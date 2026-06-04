# Sentiment Analysis Script for Survey Responses

This Python tool used OpenAI's API to perform sentiment analysis on open-ended survey responses, classifying them according to predefined sentiment categories. It was created in June 2025 with LLM support.

## Overview

The script analyzes three types of open-ended responses per respondent:
- **OETopic**: Topic preferences for entertainment content
- **OETopMD**: Most interesting story themes
- **OEUniqueMD**: Unique and different content themes

Each response is classified using sentiment codes (4, 5, 6, 7, 8, 9) that map to specific emotional and thematic categories.

## Sentiment Categories

- **Code 4 (teach_me)**: "I believe this could teach me something"
- **Code 5 (teach_others)**: "I believe this could teach other people something"
- **Code 6 (shift_my_feelings)**: "It would make me feel more understood and less alone"
- **Code 7 (shift_others_feelings)**: "It could change how other people feel"
- **Code 8 (societal_change)**: "This could change culture, politics, or society"
- **Code 9 (authenticity)**: "This is a more authentic depiction"

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key:**
   ```bash
   # Option 1: Environment variable (recommended)
   export OPENAI_API_KEY="your-api-key-here"
   
   # Option 2: Windows PowerShell
   $env:OPENAI_API_KEY="your-api-key-here"
   
   # Option 3: Pass via command line argument
   ```

3. **Ensure required files are present:**
   - `MarketcastOpensOnly.csv` - Main survey data
   - `HumanClassification.csv` - Few-shot training examples
   - `sentiment_map.json` - Sentiment category definitions

## Usage

### Basic Usage

Analyze a specific range of rows:

```bash
python ImpactAPIcalls.py --start-row 1 --end-row 100
```

### Advanced Usage

```bash
python ImpactAPIcalls.py \
    --start-row 1 \
    --end-row 100 \
    --output-file "my_results.csv" \
    --model "gpt-4" \
    --api-key "your-api-key"
```

### Command Line Arguments

- `--start-row`: Starting row number (1-indexed, required)
- `--end-row`: Ending row number (1-indexed, required)
- `--output-file`: Custom output CSV file path (optional)
- `--api-key`: OpenAI API key (optional if set as environment variable)
- `--model`: OpenAI model to use (default: gpt-4)

## Output

The script generates two files:

1. **CSV Results File**: Contains original responses plus sentiment labels
   - `uuid`: Unique respondent identifier
   - `OETopic`, `OETopMD`, `OEUniqueMD`: Original responses
   - `OETopic_labels`, `OETopMD_labels`, `OEUniqueMD_labels`: Sentiment codes

2. **Summary JSON File**: Contains analysis statistics
   - Total respondents analyzed
   - Sentiment code counts across all responses
   - Breakdown by question type

## Features

- **Few-shot Learning**: Uses human-classified examples to improve accuracy
- **Rate Limiting**: Built-in delays to respect OpenAI API limits
- **Error Handling**: Graceful handling of API errors and malformed data
- **Logging**: Comprehensive logging to both file and console
- **Batch Processing**: Process large datasets in manageable chunks
- **Progress Tracking**: Regular updates on processing status

## Example Output

### CSV Results
```csv
uuid,OETopic,OETopMD,OEUniqueMD,OETopic_labels,OETopMD_labels,OEUniqueMD_labels
5spv80y16yxyq70p,"I'd like to see a teenager in a Hispanic family navigate life...","This is a relatable theme...","I think this is most unique...","6, 9","6, 9","9"
```

### Summary JSON
```json
{
  "total_respondents": 100,
  "sentiment_counts": {
    "6": 45,
    "9": 38,
    "4": 12,
    "7": 8
  },
  "question_breakdown": {
    "OETopic": {"6": 20, "9": 15},
    "OETopMD": {"6": 18, "9": 16},
    "OEUniqueMD": {"9": 7, "6": 7}
  }
}
```

## Processing Large Datasets

For the full dataset of ~1310 respondents, consider processing in batches:

```bash
# Process first 100 respondents
python ImpactAPIcalls.py --start-row 1 --end-row 100

# Process next 100 respondents
python ImpactAPIcalls.py --start-row 101 --end-row 200

# Continue until complete
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `OPENAI_API_KEY` environment variable is set
2. **File Not Found**: Check that all required CSV and JSON files are in the same directory
3. **Rate Limiting**: The script includes delays, but you may need to increase them for high-volume usage
4. **Memory Issues**: For very large datasets, consider processing in smaller batches

### Logs

Check `sentiment_analysis.log` for detailed execution information and error messages.

## License

This script is provided as-is for research and analysis purposes.
