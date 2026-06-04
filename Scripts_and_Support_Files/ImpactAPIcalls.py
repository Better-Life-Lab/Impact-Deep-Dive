import pandas as pd
import json
import openai
import time
import logging
from typing import List, Dict, Any
import os
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the sentiment analyzer with OpenAI API key and model.
        
        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI model to use (default: gpt-4)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.sentiment_map = self._load_sentiment_map()
        self.few_shot_examples = self._load_few_shot_examples()
        
    def _load_sentiment_map(self) -> Dict[str, Dict[str, Any]]:
        """Load the sentiment mapping from JSON file."""
        try:
            with open('sentiment_map.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("sentiment_map.json not found!")
            raise
        except json.JSONDecodeError:
            logger.error("Invalid JSON in sentiment_map.json!")
            raise
    
    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples from HumanClassification.csv."""
        try:
            df = pd.read_csv('HumanClassification.csv')
            examples = []
            
            for _, row in df.iterrows():
                if pd.notna(row['OETopic_labels']) or pd.notna(row['OETopMD_labels']) or pd.notna(row['OEUniqueMD_labels']):
                    example = {
                        'uuid': row['uuid'],
                        'responses': {},
                        'labels': {}
                    }
                    
                    # Get the corresponding responses from the main data
                    main_df = pd.read_csv('MarketcastOpensOnly.csv')
                    main_row = main_df[main_df['uuid'] == row['uuid']]
                    
                    if not main_row.empty:
                        example['responses'] = {
                            'OETopic': main_row.iloc[0]['OETopic'] if pd.notna(main_row.iloc[0]['OETopic']) else '',
                            'OETopMD': main_row.iloc[0]['OETopMD'] if pd.notna(main_row.iloc[0]['OETopMD']) else '',
                            'OEUniqueMD': main_row.iloc[0]['OEUniqueMD'] if pd.notna(main_row.iloc[0]['OEUniqueMD']) else ''
                        }
                        
                        example['labels'] = {
                            'OETopic': str(row['OETopic_labels']) if pd.notna(row['OETopic_labels']) else '',
                            'OETopMD': str(row['OETopMD_labels']) if pd.notna(row['OETopMD_labels']) else '',
                            'OEUniqueMD': str(row['OEUniqueMD_labels']) if pd.notna(row['OEUniqueMD_labels']) else ''
                        }
                        
                        examples.append(example)
            
            logger.info(f"Loaded {len(examples)} few-shot examples")
            return examples
            
        except FileNotFoundError:
            logger.error("HumanClassification.csv not found!")
            return []
        except Exception as e:
            logger.error(f"Error loading few-shot examples: {e}")
            return []
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the OpenAI API."""
        sentiment_descriptions = []
        for code, details in self.sentiment_map.items():
            sentiment_descriptions.append(
                f"Code {code} ({details['name']}): {details['definition']}."
            )
        
        system_prompt = f"""You are a sentiment analysis expert. Your task is to analyze open-ended survey responses and classify them according to the following sentiment categories:

{chr(10).join(sentiment_descriptions)}

For each response, you must:
1. Read the text carefully
2. Identify which sentiment codes (4, 5, 6, 7, 8, 9) apply
3. Return ONLY the codes as a comma-separated list (e.g., "6, 9")
4. If no codes apply, return an empty string
5. You can assign multiple codes if multiple sentiments are present

IMPORTANT: Apply different interpretation standards based on question type:

- OETopic (topic preferences): Most strict - only assign if the response explicitly mentions potential impact or accuracy/realism
- OETopMD (most interesting) and OEUniqueMD (unique/different): Moderate - look for specific references to the sentiment. Avoid overinterpreting short responses that list examples without stating thoughts/feelings/behavior or accuracy/realism.

Labeling policy:
- Require explicit textual evidence for any assigned code; do not infer from topic/genre alone.
- Do not rely on keywords or paraphrases; use semantic intent and context. The presence of any single phrase is not sufficient.
- If the meaning is ambiguous or only implied, return an empty string."""
        
        return system_prompt
    
    def _create_few_shot_prompt(self) -> str:
        """Create few-shot examples for the prompt."""
        if not self.few_shot_examples:
            return ""
        
        examples_text = "Here are some examples of how to classify responses:\n\n"
        
        for example in self.few_shot_examples:  # Use all examples
            examples_text += f"UUID: {example['uuid']}\n"
            for question, response in example['responses'].items():
                if response.strip():
                    examples_text += f"{question}: {response}\n"
                    examples_text += f"Labels: {example['labels'][question]}\n\n"
        
        return examples_text
    
    def analyze_response(self, response_text: str, question_type: str) -> str:
        """
        Analyze a single response using OpenAI API.
        
        Args:
            response_text (str): The text response to analyze
            question_type (str): The type of question (OETopic, OETopMD, OEUniqueMD)
            
        Returns:
            str: Comma-separated sentiment codes
        """
        if not response_text or response_text.strip() == '':
            return ""
        
        system_prompt = self._create_system_prompt()
        few_shot_examples = self._create_few_shot_prompt()
        
        user_prompt = f"""{few_shot_examples}
Now analyze this response for question type '{question_type}':

Response: {response_text}

Please provide the sentiment codes that apply to this response:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Stricter, deterministic outputs
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"Analysis result for {question_type}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing response: {e}")
            return ""
    
    def analyze_respondent(self, row: pd.Series) -> Dict[str, Any]:
        """
        Analyze all three responses for a single respondent.
        
        Args:
            row (pd.Series): Row from the main dataframe
            
        Returns:
            Dict containing analysis results
        """
        uuid = row['uuid']
        logger.info(f"Analyzing respondent: {uuid}")
        
        results = {
            'uuid': uuid,
            'OETopic': row['OETopic'] if pd.notna(row['OETopic']) else '',
            'OETopMD': row['OETopMD'] if pd.notna(row['OETopMD']) else '',
            'OEUniqueMD': row['OEUniqueMD'] if pd.notna(row['OEUniqueMD']) else '',
            'OETopic_labels': '',
            'OETopMD_labels': '',
            'OEUniqueMD_labels': ''
        }
        
        # Analyze each response
        for question in ['OETopic', 'OETopMD', 'OEUniqueMD']:
            if results[question].strip():
                labels = self.analyze_response(results[question], question)
                results[f'{question}_labels'] = labels
                
                # Add delay to respect API rate limits
                time.sleep(1.0)  # 1 second between calls to avoid 429s
        
        return results
    
    def analyze_batch(self, start_row: int, end_row: int, output_file: str = None) -> pd.DataFrame:
        """
        Analyze a batch of respondents.
        
        Args:
            start_row (int): Starting row number (1-indexed)
            end_row (int): Ending row number (1-indexed)
            output_file (str): Optional output file path
            
        Returns:
            pd.DataFrame: DataFrame with analysis results
        """
        # Load the main data
        df = pd.read_csv('MarketcastOpensOnly.csv')
        
        # Adjust for 1-indexed input
        start_idx = start_row - 1
        end_idx = end_row
        
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(df):
            end_idx = len(df)
        
        logger.info(f"Analyzing rows {start_row} to {end_row} (indices {start_idx} to {end_idx})")
        
        batch_df = df.iloc[start_idx:end_idx].copy()
        results = []
        
        for idx, row in batch_df.iterrows():
            try:
                result = self.analyze_respondent(row)
                results.append(result)
                
                # Progress update every 10 respondents
                if len(results) % 10 == 0:
                    logger.info(f"Processed {len(results)} respondents...")
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Add empty result for failed rows
                results.append({
                    'uuid': row['uuid'],
                    'OETopic': row['OETopic'] if pd.notna(row['OETopic']) else '',
                    'OETopMD': row['OETopMD'] if pd.notna(row['OETopMD']) else '',
                    'OEUniqueMD': row['OEUniqueMD'] if pd.notna(row['OEUniqueMD']) else '',
                    'OETopic_labels': 'ERROR',
                    'OETopMD_labels': 'ERROR',
                    'OEUniqueMD_labels': 'ERROR'
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        if output_file:
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return results_df
    
    def get_sentiment_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of sentiment analysis results.
        
        Args:
            results_df (pd.DataFrame): DataFrame with analysis results
            
        Returns:
            Dict containing summary statistics
        """
        summary = {
            'total_respondents': len(results_df),
            'sentiment_counts': {},
            'question_breakdown': {}
        }
        
        # Count sentiment codes across all questions
        all_labels = []
        for question in ['OETopic_labels', 'OETopMD_labels', 'OEUniqueMD_labels']:
            labels = results_df[question].dropna()
            for label_str in labels:
                if label_str and label_str != 'ERROR':
                    codes = [code.strip() for code in label_str.split(',')]
                    all_labels.extend(codes)
        
        # Count occurrences of each sentiment code
        for code in all_labels:
            if code in self.sentiment_map:
                summary['sentiment_counts'][code] = summary['sentiment_counts'].get(code, 0) + 1
        
        # Breakdown by question
        for question in ['OETopic', 'OETopMD', 'OEUniqueMD']:
            question_labels = results_df[f'{question}_labels'].dropna()
            question_summary = {}
            
            for label_str in question_labels:
                if label_str and label_str != 'ERROR':
                    codes = [code.strip() for code in label_str.split(',')]
                    for code in codes:
                        if code in self.sentiment_map:
                            question_summary[code] = question_summary.get(code, 0) + 1
            
            summary['question_breakdown'][question] = question_summary
        
        return summary

def main():
    """Main function to run the sentiment analysis."""
    parser = argparse.ArgumentParser(description='Run sentiment analysis on survey responses')
    parser.add_argument('--start-row', type=int, required=True, help='Starting row number (1-indexed)')
    parser.add_argument('--end-row', type=int, required=True, help='Ending row number (1-indexed)')
    parser.add_argument('--output-file', type=str, help='Output CSV file path')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model to use')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        return
    
    # Create output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"sentiment_analysis_{args.start_row}_{args.end_row}_{timestamp}.csv"
    
    try:
        # Initialize analyzer
        analyzer = SentimentAnalyzer(api_key, args.model)
        
        # Run analysis
        logger.info(f"Starting sentiment analysis for rows {args.start_row} to {args.end_row}")
        results_df = analyzer.analyze_batch(args.start_row, args.end_row, args.output_file)
        
        # Generate summary
        summary = analyzer.get_sentiment_summary(results_df)
        
        # Print summary
        logger.info("Analysis complete!")
        logger.info(f"Total respondents analyzed: {summary['total_respondents']}")
        logger.info("Sentiment code counts:")
        for code, count in summary['sentiment_counts'].items():
            name = analyzer.sentiment_map[code]['name']
            logger.info(f"  Code {code} ({name}): {count}")
        
        # Save summary to JSON
        summary_file = args.output_file.replace('.csv', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
