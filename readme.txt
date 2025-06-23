# News Article Semantic Analysis Tool

A comprehensive Python tool for analyzing German news articles using OpenAI's GPT API. Features automatic stop word removal, sentiment analysis, political stance detection, and structured CSV output.

## Features

- **Multi-format Support**: Process both `.txt` and `.pdf` files
- **German Stop Word Filtering**: Automatic removal of German stop words
- **Semantic Analysis**: Sentiment, political stance, bias detection, and more
- **Structured Output**: CSV and JSON results with comprehensive logging
- **Cost Tracking**: Real-time API cost estimation
- **Error Handling**: Robust retry logic and detailed error reporting
- **Timestamped Results**: Each run creates a separate output folder

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv semantic_analysis_env

# Activate environment (Windows)
semantic_analysis_env\Scripts\activate

# Activate environment (macOS/Linux)
source semantic_analysis_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy environment template
cp .env.template .env

# Create .env textfile and paste OPENAI_API_KEY=your_openai_api_key_here 
# Get your key from: https://platform.openai.com/api-keys
```

### 3. Prepare Articles

```bash
# Create articles folder
mkdir articles

# Add your .txt or .pdf files to the articles folder
```

### 4. Run Analysis

```bash
python gpt_analyse_7.py
```

## Configuration

Edit `config.py` to customize:

- **GPT Model**: Choose between `gpt-4o-mini` (fast/cheap) or `gpt-4o` (high quality)
- **Analysis Fields**: Customize what aspects to analyze
- **File Processing**: Set size limits and supported formats
- **Output Format**: Configure CSV columns and file names

### Quick Configuration Presets

```python
# In config.py, uncomment for quick analysis:
# config_preset = QUICK_CONFIG
# for key, value in config_preset.items():
#     globals()[key] = value
```

## Output Files

Each analysis run creates a timestamped folder with:

- `analysis_results.csv` - Main results (open in Excel/Google Sheets)
- `detailed_results.json` - Complete analysis data
- `analysis_summary.txt` - Human-readable summary
- `analysis.log` - Processing log
- `german_stopwords.txt` - Stop words used (if enabled)

## Analysis Fields

The tool analyzes each article for:

- **Main Topic**: Brief 2-4 word topic summary
- **Sentiment**: Positive/Negative/Neutral
- **Political Stance**: Left/Center-Left/Center/Center-Right/Right/Neutral
- **Key Entities**: Important people, places, organizations
- **Keywords**: 5-8 most important terms
- **Summary**: 2-3 sentence German summary
- **Bias Level**: Low/Medium/High
- **Credibility**: High/Medium/Low
- **Target Audience**: General/Specialized/Political/Business/etc.
- **Article Type**: News/Opinion/Analysis/Report/Interview/etc.

## Cost Estimation

The tool provides real-time cost estimates:

- **gpt-4o-mini**: ~$0.15-0.60 per 1M tokens (recommended)
- **gpt-4o**: ~$2.50-10.00 per 1M tokens (higher quality)

Typical cost per article: $0.001-0.01 (depending on length and model)

## Troubleshooting

### Common Issues

1. **"config.py not found"**
   - Ensure `config.py` is in the same directory as the script

2. **"API key not found"**
   - Check your `.env` file contains: `OPENAI_API_KEY=your_key_here`
   - Ensure the environment is activated

3. **"No supported files found"**
   - Check the `articles` folder contains `.txt` or `.pdf` files
   - Verify `ARTICLES_FOLDER` path in `config.py`

4. **"File too large"**
   - Increase `MAX_FILE_SIZE_MB` in `config.py`
   - Or split large files into smaller chunks

### Debug Mode

Enable detailed logging by setting `DEBUG_MODE = True` in `config.py`:

```python
DEBUG_MODE = True  # Enable verbose logging
```

## Customization

### Custom Analysis Prompts

Modify `ANALYSIS_PROMPT` in `config.py` to change what the AI analyzes:

```python
ANALYSIS_PROMPT = """
Your custom analysis instructions here...
"""
```

### Alternative Models

```python
# For faster, cheaper analysis:
GPT_MODEL = "gpt-4o-mini"

# For higher quality analysis:
GPT_MODEL = "gpt-4o"
```

### Disable Stop Word Filtering

```python
ENABLE_STOPWORD_FILTERING = False
```

## File Structure

```
project/
├── gpt_analyse_7.py      # Main script
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── .env                  # API keys (create from .env.template)
├── articles/             # Input articles (.txt, .pdf)
└── analysis_results/     # Output folders (auto-created)
    └── analysis_20241223_143022/
        ├── analysis_results.csv
        ├── detailed_results.json
        ├── analysis_summary.txt
        └── analysis.log
```

## Security Notes

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure
- Monitor your API usage and costs

## Support

For issues or questions:

1. Check the log file in your output directory
2. Enable `DEBUG_MODE` for detailed information
3. Verify your API key and internet connection
4. Ensure all dependencies are installed correctly

## License

This tool is for educational and research purposes. Please respect OpenAI's usage policies and any applicable data privacy regulations when processing news articles.
