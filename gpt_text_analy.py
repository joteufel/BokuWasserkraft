#!/usr/bin/env python3
"""
News Article Semantic Analysis Script with CSV Output

Usage:
source semantic_analysis_env/bin/activate
python gpt_analyse_7.py
deactivate  # when done

This script processes news articles from txt/pdf files, removes German stop words,
and performs semantic analysis using OpenAI's GPT API.

Features:
- Separate configuration file (config.py)
- Timestamped output folders
- CSV output with structured data
- Comprehensive error handling
- Progress tracking and logging
- Cost estimation

Requirements:
- pip install openai PyPDF2 requests python-dotenv pandas

Configuration:
- Edit config.py to customize settings
- Set your OpenAI API key in .env file or environment variable
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
import json
import traceback
from datetime import datetime
import csv
import re

# External libraries
import requests
from openai import OpenAI
import PyPDF2
from dotenv import load_dotenv

# Import configuration
try:
    import config
except ImportError:
    print("ERROR: config.py file not found.")
    print("Please ensure config.py is in the same directory as this script.")
    sys.exit(1)

# Try to import pandas for enhanced CSV handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

def create_output_directory(base_dir: str = None) -> Path:
    """
    Create a timestamped output directory for this analysis run.
    
    Args:
        base_dir (str): Base directory for output folders
        
    Returns:
        Path: Created output directory path
    """
    base_dir = base_dir or config.BASE_OUTPUT_DIR
    
    # Create timestamp for folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"analysis_{timestamp}"
    
    # Create base directory if it doesn't exist
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subdirectory
    output_dir = base_path / run_id
    output_dir.mkdir(exist_ok=True)
    
    return output_dir

def setup_logging(output_dir: Path, debug: bool = None) -> logging.Logger:
    """
    Set up logging with file output in the analysis directory.
    
    Args:
        output_dir (Path): Directory to save log file
        debug (bool): Enable debug level logging
        
    Returns:
        logging.Logger: Configured logger
    """
    debug = debug if debug is not None else config.DEBUG_MODE
    
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = output_dir / config.LOG_FILENAME
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

class CSVResultParser:
    """Helper class to parse GPT responses into structured CSV data."""
    
    @staticmethod
    def parse_gpt_response(response_text: str) -> Dict[str, str]:
        """
        Parse structured GPT response into a dictionary.
        
        Args:
            response_text (str): GPT response text
            
        Returns:
            Dict[str, str]: Parsed fields
        """
        # Initialize fields with empty defaults
        fields = {field: '' for field in config.FIELD_MAPPING.values()}
        
        # Parse the response line by line
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                # Split on first colon only
                field_name, field_value = line.split(':', 1)
                field_name = field_name.strip().lower()
                field_value = field_value.strip()
                
                # Map field names to our standard names
                if field_name in config.FIELD_MAPPING:
                    fields[config.FIELD_MAPPING[field_name]] = field_value
        
        # Clean up fields
        for key, value in fields.items():
            # Remove extra whitespace and quotes
            fields[key] = value.strip().strip('"').strip("'")
            
            # Convert semicolon-separated lists to cleaner format
            if key in config.LIST_FIELDS and value:
                items = [item.strip() for item in value.split(';') if item.strip()]
                fields[key] = '; '.join(items)
        
        return fields

class NewsArticleAnalyzer:
    """Main class for processing and analyzing news articles."""
    
    def __init__(self, api_key: str, output_dir: Path):
        """
        Initialize the analyzer.
        
        Args:
            api_key (str): OpenAI API key
            output_dir (Path): Directory for saving outputs
        """
        self.client = OpenAI(api_key=api_key)
        self.output_dir = output_dir
        self.stop_words = set()
        self.csv_parser = CSVResultParser()
        
        # Create logger (should already be set up by main)
        self.logger = logging.getLogger(__name__)
        
        if config.DEBUG_MODE:
            self.logger.debug("Debug mode enabled")
            
        self.logger.info(f"Initialized analyzer with model: {config.GPT_MODEL}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Stop word filtering: {'enabled' if config.ENABLE_STOPWORD_FILTERING else 'disabled'}")
    
    def load_german_stopwords(self) -> bool:
        """
        Load German stop words from the configured URL.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not config.ENABLE_STOPWORD_FILTERING:
            self.logger.info("Stop word filtering disabled, skipping download")
            return True
            
        try:
            self.logger.info(f"Downloading German stop words from: {config.STOPWORDS_URL}")
            
            response = requests.get(config.STOPWORDS_URL, timeout=config.API_TIMEOUT)
            response.raise_for_status()
            
            # Parse stop words (one per line)
            stop_words_text = response.text.strip()
            self.stop_words = set(word.strip().lower() for word in stop_words_text.split('\n') if word.strip())
            
            # Save stop words to output directory for reference
            stopwords_file = self.output_dir / config.STOPWORDS_FILENAME
            with open(stopwords_file, 'w', encoding='utf-8') as f:
                for word in sorted(self.stop_words):
                    f.write(f"{word}\n")
            
            self.logger.info(f"Loaded {len(self.stop_words)} German stop words")
            self.logger.info(f"Stop words saved to: {stopwords_file}")
            
            if config.DEBUG_MODE:
                self.logger.debug(f"Sample stop words: {list(self.stop_words)[:10]}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download stop words: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error loading stop words: {e}")
            if config.DEBUG_MODE:
                self.logger.debug(traceback.format_exc())
            return False
    
    def extract_text_from_file(self, file_path: Path) -> Optional[str]:
        """
        Extract text from txt or pdf file.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            Optional[str]: Extracted text or None if failed
        """
        try:
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                self.logger.warning(f"File too large ({file_size_mb:.1f}MB): {file_path}")
                return None
            
            if file_path.suffix.lower() == '.txt':
                return self._extract_from_txt(file_path)
            elif file_path.suffix.lower() == '.pdf':
                return self._extract_from_pdf(file_path)
            else:
                self.logger.warning(f"Unsupported file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to extract text from {file_path}: {e}")
            if config.DEBUG_MODE:
                self.logger.debug(traceback.format_exc())
            return None
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from txt file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            self.logger.debug(f"Extracted {len(text)} characters from {file_path}")
            return text
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
            self.logger.debug(f"Extracted {len(text)} characters from {file_path} (latin-1)")
            return text
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from pdf file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num} of {file_path}: {e}")
        
        self.logger.debug(f"Extracted {len(text)} characters from {file_path}")
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove German stop words from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stop words removed (or original if filtering disabled)
        """
        if not config.ENABLE_STOPWORD_FILTERING:
            self.logger.debug("Stop word filtering disabled, returning original text")
            return text
            
        if not self.stop_words:
            self.logger.warning("No stop words loaded, returning original text")
            return text
        
        # Simple word tokenization and filtering
        words = text.split()
        filtered_words = [word for word in words if word.lower().strip('.,!?;:"()[]{}') not in self.stop_words]
        
        filtered_text = ' '.join(filtered_words)
        
        if config.DEBUG_MODE:
            self.logger.debug(f"Original word count: {len(words)}")
            self.logger.debug(f"Filtered word count: {len(filtered_words)}")
            self.logger.debug(f"Removed {len(words) - len(filtered_words)} stop words")
        
        return filtered_text
    
    def analyze_with_gpt(self, text: str, prompt: str) -> Optional[Dict]:
        """
        Analyze text using GPT API with custom prompt.
        
        Args:
            text (str): Text to analyze
            prompt (str): Analysis prompt
            
        Returns:
            Optional[Dict]: Analysis results or None if failed
        """
        # For CSV output, we'll handle large texts by summarizing first if needed
        if len(text) > config.MAX_CHUNK_SIZE:
            self.logger.info(f"Text too long ({len(text)} chars), creating summary first...")
            text = self._create_summary_for_long_text(text)
        
        full_prompt = f"{prompt}\n\nText to analyze:\n{text}"
        
        for attempt in range(config.MAX_RETRIES):
            try:
                self.logger.info(f"Sending text to GPT (attempt {attempt + 1}/{config.MAX_RETRIES})")
                
                response = self.client.chat.completions.create(
                    model=config.GPT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in semantic text analysis. Always follow the requested output format exactly."},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=config.GPT_MAX_TOKENS,
                    temperature=config.GPT_TEMPERATURE,
                    top_p=config.GPT_TOP_P
                )
                
                usage = response.usage
                estimated_cost = config.get_estimated_cost(
                    usage.prompt_tokens, 
                    usage.completion_tokens, 
                    config.GPT_MODEL
                )
                
                analysis_result = {
                    "model": config.GPT_MODEL,
                    "prompt": prompt,
                    "response": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    },
                    "estimated_cost": estimated_cost
                }
                
                self.logger.info("Analysis completed successfully")
                if config.DEBUG_MODE:
                    self.logger.debug(f"Token usage: {analysis_result['usage']}")
                    self.logger.debug(f"Estimated cost: ${estimated_cost:.6f}")
                
                return analysis_result
                
            except Exception as e:
                self.logger.error(f"GPT API error (attempt {attempt + 1}): {e}")
                if attempt < config.MAX_RETRIES - 1:
                    wait_time = config.RETRY_BASE_DELAY * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Max retries reached")
                    if config.DEBUG_MODE:
                        self.logger.debug(traceback.format_exc())
        
        return None
    
    def _create_summary_for_long_text(self, text: str) -> str:
        """Create a summary of long text for analysis."""
        try:
            summary_prompt = config.SUMMARY_PROMPT_TEMPLATE.format(
                max_words=config.MAX_SUMMARY_WORDS,
                text=text[:config.MAX_CHUNK_SIZE]
            )
            
            response = self.client.chat.completions.create(
                model=config.GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates accurate summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=config.MAX_SUMMARY_WORDS + 100,
                temperature=config.GPT_TEMPERATURE
            )
            
            summary = response.choices[0].message.content
            self.logger.info(f"Created summary: {len(text)} chars -> {len(summary)} chars")
            return summary
            
        except Exception as e:
            self.logger.warning(f"Failed to create summary, using truncated text: {e}")
            return text[:config.MAX_CHUNK_SIZE]
    
    def process_folder(self, folder_path: str, prompt: str) -> List[Dict]:
        """
        Process all articles in a folder.
        
        Args:
            folder_path (str): Path to folder containing articles
            prompt (str): Analysis prompt
            
        Returns:
            List[Dict]: List of analysis results
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all supported files
        files = []
        for ext in config.SUPPORTED_EXTENSIONS:
            files.extend(list(folder.glob(f"*{ext}")))
        
        if not files:
            self.logger.warning(f"No supported files found in {folder_path}")
            self.logger.info(f"Supported extensions: {', '.join(config.SUPPORTED_EXTENSIONS)}")
            return []
        
        self.logger.info(f"Found {len(files)} files to process")
        
        results = []
        
        for i, file_path in enumerate(files, 1):
            self.logger.info(f"Processing file {i}/{len(files)}: {file_path.name}")
            
            try:
                # Extract texts
                raw_text = self.extract_text_from_file(file_path)
                if not raw_text:
                    self.logger.error(f"Failed to extract text from {file_path}")
                    continue
                
                # Check minimum file size
                if len(raw_text) < config.MIN_FILE_SIZE:
                    self.logger.warning(f"File too small ({len(raw_text)} chars), skipping: {file_path}")
                    continue
                
                # Remove stop words
                filtered_text = self.remove_stopwords(raw_text)
                
                # Analyze with GPT
                analysis = self.analyze_with_gpt(filtered_text, prompt)
                
                if analysis:
                    # Parse the GPT response into structured data
                    parsed_data = self.csv_parser.parse_gpt_response(analysis['response'])
                    
                    result = {
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "file_size_chars": len(raw_text),
                        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "tokens_used": analysis['usage']['total_tokens'],
                        "estimated_cost": analysis['estimated_cost'],
                        "model": analysis['model'],
                        **parsed_data,  # Merge the parsed analysis data
                        "raw_analysis": analysis  # Keep the raw analysis for detailed JSON
                    }
                    results.append(result)
                    self.logger.info(f"Successfully processed {file_path.name} (${analysis['estimated_cost']:.6f})")
                else:
                    self.logger.error(f"Failed to analyze {file_path.name}")
                
                # Add delay between files
                if i < len(files):
                    time.sleep(config.DELAY_BETWEEN_FILES)
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                if config.DEBUG_MODE:
                    self.logger.debug(traceback.format_exc())
                continue
        
        return results
    
    def save_results_as_csv(self, results: List[Dict]) -> None:
        """
        Save analysis results as CSV file.
        
        Args:
            results (List[Dict]): Analysis results to save
        """
        if not results:
            self.logger.warning("No results to save")
            return
        
        csv_file = self.output_dir / config.CSV_FILENAME
        
        try:
            # If pandas is available, use it for better CSV handling
            if PANDAS_AVAILABLE:
                # Create DataFrame
                df_data = []
                for result in results:
                    row = {col: result.get(col, '') for col in config.CSV_COLUMNS}
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_csv(csv_file, index=False, encoding='utf-8')
                self.logger.info(f"CSV results saved to {csv_file} (using pandas)")
            else:
                # Fallback to standard CSV library
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=config.CSV_COLUMNS)
                    writer.writeheader()
                    for result in results:
                        row = {col: result.get(col, '') for col in config.CSV_COLUMNS}
                        writer.writerow(row)
                self.logger.info(f"CSV results saved to {csv_file} (using csv library)")
            
            # Also save detailed JSON for reference
            json_file = self.output_dir / config.DETAILED_JSON_FILENAME
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Detailed JSON saved to {json_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save CSV results: {e}")
    
    def create_run_summary(self, results: List[Dict]) -> None:
        """
        Create a human-readable summary of the analysis run.
        
        Args:
            results (List[Dict]): Analysis results
        """
        summary_file = self.output_dir / config.SUMMARY_FILENAME
        
        try:
            total_files = len(results)
            total_tokens = sum(r.get("tokens_used", 0) for r in results)
            total_cost = sum(r.get("estimated_cost", 0) for r in results)
            
            # Analysis statistics
            sentiments = [r.get('sentiment', '').lower() for r in results if r.get('sentiment')]
            sentiment_counts = {
                'positive': sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')
            }
            
            political_stances = [r.get('political_stance', '').lower() for r in results if r.get('political_stance')]
            stance_counts = {}
            for stance in political_stances:
                stance_counts[stance] = stance_counts.get(stance, 0) + 1
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== NEWS ARTICLE ANALYSIS SUMMARY ===\n\n")
                f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"GPT Model used: {config.GPT_MODEL}\n")
                f.write(f"Stop word filtering: {'enabled' if config.ENABLE_STOPWORD_FILTERING else 'disabled'}\n\n")
                
                f.write("--- PROCESSING RESULTS ---\n")
                f.write(f"Files processed: {total_files}\n")
                f.write(f"Total tokens used: {total_tokens:,}\n")
                f.write(f"Total estimated cost: ${total_cost:.6f}\n")
                f.write(f"Average cost per file: ${total_cost/total_files:.6f}\n\n" if total_files > 0 else "")
                
                f.write("--- ANALYSIS OVERVIEW ---\n")
                f.write(f"Sentiment Distribution:\n")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / total_files * 100) if total_files > 0 else 0
                    f.write(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)\n")
                
                if stance_counts:
                    f.write(f"\nPolitical Stance Distribution:\n")
                    for stance, count in sorted(stance_counts.items()):
                        percentage = (count / total_files * 100) if total_files > 0 else 0
                        f.write(f"  {stance.replace('-', ' ').title()}: {count} ({percentage:.1f}%)\n")
                
                f.write("\n--- FILES ANALYZED ---\n")
                for i, result in enumerate(results, 1):
                    f.write(f"{i:2}. {result['file_name']}\n")
                    f.write(f"    Topic: {result.get('main_topic', 'N/A')}\n")
                    f.write(f"    Sentiment: {result.get('sentiment', 'N/A')}\n")
                    f.write(f"    Tokens: {result.get('tokens_used', 0):,}\n")
                    f.write(f"    Cost: ${result.get('estimated_cost', 0):.6f}\n\n")
                
                f.write("--- OUTPUT FILES ---\n")
                f.write(f"• {config.CSV_FILENAME} - Main CSV results (open in Excel/Google Sheets)\n")
                f.write(f"• {config.DETAILED_JSON_FILENAME} - Detailed JSON backup\n")
                f.write(f"• {config.SUMMARY_FILENAME} - This summary\n")
                f.write(f"• {config.LOG_FILENAME} - Processing log\n")
                if config.ENABLE_STOPWORD_FILTERING:
                    f.write(f"• {config.STOPWORDS_FILENAME} - Stop words used\n")
                
                f.write("\n--- CONFIGURATION USED ---\n")
                f.write(f"• Model: {config.GPT_MODEL}\n")
                f.write(f"• Max chunk size: {config.MAX_CHUNK_SIZE:,} characters\n")
                f.write(f"• Stop word filtering: {config.ENABLE_STOPWORD_FILTERING}\n")
                f.write(f"• Debug mode: {config.DEBUG_MODE}\n")
            
            self.logger.info(f"Run summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create run summary: {e}")


def validate_environment():
    """Validate environment and dependencies."""
    errors = []
    
    # Check for API key
    api_key = os.getenv(config.OPENAI_API_KEY_ENV)
    if not api_key:
        errors.append(f"OpenAI API key not found in environment variable '{config.OPENAI_API_KEY_ENV}'")
    
    # Check if articles folder exists
    if not Path(config.ARTICLES_FOLDER).exists():
        errors.append(f"Articles folder does not exist: {config.ARTICLES_FOLDER}")
    
    # Validate configuration
    try:
        config.validate_config()
    except ValueError as e:
        errors.append(str(e))
    
    if errors:
        print("❌ Environment validation failed:")
        for error in errors:
            print(f"   • {error}")
        print(f"\nPlease fix these issues before running the script.")
        print(f"Create a .env file with: {config.OPENAI_API_KEY_ENV}=your_api_key_here")
        return False
    
    return True


def main():
    """Main function with automatic output directory creation."""
    
    print("🚀 News Article Semantic Analysis Tool")
    print("=====================================")
    
    # Load environment variables
    load_dotenv()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Get API key from environment
    api_key = os.getenv(config.OPENAI_API_KEY_ENV)
    
    try:
        # Create timestamped output directory
        output_dir = create_output_directory()
        print(f"📁 Created output directory: {output_dir}")
        
        # Set up logging in the output directory
        logger = setup_logging(output_dir)
        
        # Initialize analyzer
        analyzer = NewsArticleAnalyzer(api_key, output_dir)
        
        # Load German stop words
        logger.info("Loading German stop words...")
        if not analyzer.load_german_stopwords():
            logger.warning("Failed to load stop words, continuing without filtering")
        
        # Process articles
        logger.info(f"Starting analysis of articles in: {config.ARTICLES_FOLDER}")
        logger.info(f"Using analysis prompt: {config.ANALYSIS_PROMPT}")
        
        results = analyzer.process_folder(config.ARTICLES_FOLDER, config.ANALYSIS_PROMPT)
        
        if not results:
            print("❌ No articles were successfully processed")
            logger.error("No results to save")
            return
        
        # Save results
        logger.info("Saving results...")
        analyzer.save_results_as_csv(results)
        analyzer.create_run_summary(results)
        
        # Print summary
        total_cost = sum(r.get("estimated_cost", 0) for r in results)
        total_tokens = sum(r.get("tokens_used", 0) for r in results)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Processed {len(results)} articles")
        print(f"🎯 Total tokens used: {total_tokens:,}")
        print(f"💰 Total estimated cost: ${total_cost:.6f}")
        print(f"📁 All files saved to: {output_dir}")
        print(f"\n💡 Open the CSV file in Excel, Google Sheets, or any spreadsheet program")
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if config.DEBUG_MODE:
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()