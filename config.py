"""
Configuration file for News Article Semantic Analysis Script

Modify these settings to customize the analysis behavior.
"""

# =============================================================================
# BASIC SETTINGS
# =============================================================================

# OpenAI API Settings
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"  # Environment variable name for API key
GPT_MODEL = "gpt-4o-mini"  # or "gpt-4o" for higher quality (more expensive)
GPT_MAX_TOKENS = 1000  # Maximum tokens for GPT response
GPT_TEMPERATURE = 0.3  # Lower = more deterministic, higher = more creative
GPT_TOP_P = 0.9  # Nucleus sampling parameter

# File Processing Settings
ARTICLES_FOLDER = "./Text"  # Folder containing articles to analyze
SUPPORTED_EXTENSIONS = [".txt", ".pdf"]  # File types to process
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB
MIN_FILE_SIZE = 100  # Minimum file size in characters
MAX_CHUNK_SIZE = 50000  # Maximum characters per GPT request

# Summary Settings
ENABLE_SUMMARY = True  # Set to False to disable text summarization for long articles
MAX_SUMMARY_WORDS = 800  # Maximum words for text summary

# Stop Word Filtering
ENABLE_STOPWORD_FILTERING = True  # Set to False to disable stop word removal
STOPWORDS_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-de/master/stopwords-de.txt"
STOPWORDS_FILENAME = "german_stopwords.txt"

# Processing Settings
MAX_RETRIES = 3  # Maximum API retry attempts
RETRY_BASE_DELAY = 2  # Base delay between retries (seconds)
DELAY_BETWEEN_FILES = 1  # Delay between processing files (seconds)
API_TIMEOUT = 30  # HTTP request timeout (seconds)

# Output Settings
BASE_OUTPUT_DIR = "./analysis_resultss"  # Base directory for outputs
CSV_FILENAME = "analysis_results.csv"
DETAILED_JSON_FILENAME = "detailed_results.json"
SUMMARY_FILENAME = "analysis_summary.txt"
LOG_FILENAME = "analysis.log"

# Debug Mode
DEBUG_MODE = False  # Set to True for detailed logging

# =============================================================================
# FLEXIBLE ANALYSIS CONFIGURATION
# =============================================================================

# Define analysis fields with full customization
# Each field can have: prompt_name, description, options, csv_column, enabled
ANALYSIS_FIELDS = {
    "hauptthema": {
        "enabled": True,
        "prompt_name": "Hauptthema",
        "description": "kurzes Thema in 2-4 Wörtern",
        "csv_column": "main_topic",
        "field_type": "text",
        "instructions": "Seien Sie prägnant und beschreibend"
    },
    "stimmung": {
        "enabled": True,
        "prompt_name": "Stimmung",
        "description": "positiv/negativ/neutral",
        "csv_column": "sentiment",
        "field_type": "choice",
        "options": ["positiv", "negativ", "neutral"],
        "instructions": "Bewerten Sie die Gesamtstimmung des Artikels"
    },
    "politische_haltung": {
        "enabled": True,
        "prompt_name": "Politische Haltung",
        "description": "links/mitte-links/mitte/mitte-rechts/rechts/neutral",
        "csv_column": "political_stance",
        "field_type": "choice",
        "options": ["links", "mitte-links", "mitte", "mitte-rechts", "rechts", "neutral"],
        "instructions": "Bewerten Sie die politische Ausrichtung des Artikels"
    },
    "schluesselentitaeten": {
        "enabled": True,
        "prompt_name": "Schlüsselentitäten",
        "description": "Liste getrennt durch Semikolons",
        "csv_column": "key_entities",
        "field_type": "list",
        "separator": ";",
        "instructions": "Extrahieren Sie Personen, Organisationen, Orte"
    },
    "schluesselwoerter": {
        "enabled": True,
        "prompt_name": "Schlüsselwörter",
        "description": "5-8 wichtigste Schlüsselwörter getrennt durch Semikolons",
        "csv_column": "keywords",
        "field_type": "list",
        "separator": ";",
        "instructions": "Wählen Sie die relevantesten Begriffe aus"
    },
    "zusammenfassung": {
        "enabled": True,
        "prompt_name": "Zusammenfassung",
        "description": "2-3 Sätze Zusammenfassung auf Deutsch",
        "csv_column": "summary",
        "field_type": "text",
        "instructions": "Fassen Sie die wichtigsten Punkte zusammen"
    },
    "verzerrungsgrad": {
        "enabled": True,
        "prompt_name": "Verzerrungsgrad",
        "description": "niedrig/mittel/hoch",
        "csv_column": "bias_level",
        "field_type": "choice",
        "options": ["niedrig", "mittel", "hoch"],
        "instructions": "Bewerten Sie die Objektivität des Artikels"
    },
    "glaubwuerdigkeit": {
        "enabled": True,
        "prompt_name": "Glaubwürdigkeit",
        "description": "hoch/mittel/niedrig",
        "csv_column": "credibility",
        "field_type": "choice",
        "options": ["hoch", "mittel", "niedrig"],
        "instructions": "Bewerten Sie die Vertrauenswürdigkeit der Quelle"
    },
    "zielgruppe": {
        "enabled": True,
        "prompt_name": "Zielgruppe",
        "description": "allgemein/spezialisiert/politisch/geschäftlich/etc.",
        "csv_column": "target_audience",
        "field_type": "text",
        "instructions": "Identifizieren Sie die primäre Leserschaft"
    },
    "artikeltyp": {
        "enabled": True,
        "prompt_name": "Artikeltyp",
        "description": "nachrichten/meinung/analyse/bericht/interview/etc.",
        "csv_column": "article_type",
        "field_type": "text",
        "instructions": "Klassifizieren Sie die Art des Artikels"
    },
    "argumentwasser": {
        "enabled": True,
        "prompt_name": "Argumentwasser",
        "description": "nachrichten/meinung/analyse/bericht/interview/etc.",
        "csv_column": "argument_wasser",
        "field_type": "text",
        "instructions": "Identifiziere das Hauptargument für oder gegen Wasserkraft und schreibe es in Stichworten"
    }
}

# Add custom analysis fields here
CUSTOM_ANALYSIS_FIELDS = {
    # Example custom fields (disabled by default):
    "emotionale_wirkung": {
        "enabled": False,
        "prompt_name": "Emotionale Wirkung",
        "description": "stark/mittel/schwach",
        "csv_column": "emotional_impact",
        "field_type": "choice",
        "options": ["stark", "mittel", "schwach"],
        "instructions": "Bewerten Sie die emotionale Intensität"
    },
    "komplexitaet": {
        "enabled": False,
        "prompt_name": "Komplexität",
        "description": "einfach/mittel/komplex",
        "csv_column": "complexity",
        "field_type": "choice",
        "options": ["einfach", "mittel", "komplex"],
        "instructions": "Bewerten Sie die Verständlichkeit für Laien"
    },
    "quellen_qualitaet": {
        "enabled": False,
        "prompt_name": "Quellenqualität",
        "description": "ausgezeichnet/gut/befriedigend/mangelhaft",
        "csv_column": "source_quality",
        "field_type": "choice",
        "options": ["ausgezeichnet", "gut", "befriedigend", "mangelhaft"],
        "instructions": "Bewerten Sie die Qualität der zitierten Quellen"
    }
}

# Merge custom fields with main fields
ALL_ANALYSIS_FIELDS = {**ANALYSIS_FIELDS, **CUSTOM_ANALYSIS_FIELDS}

# =============================================================================
# DYNAMIC PROMPT GENERATION
# =============================================================================

# Custom prompt header and footer
PROMPT_HEADER = """
Bitte analysieren Sie den folgenden deutschen Nachrichtenartikel und geben Sie die Ergebnisse in dem unten angegebenen Format aus. 
Jedes Feld sollte in einer separaten Zeile stehen, gefolgt von einem Doppelpunkt und dem Inhalt.

Erforderliches Format:"""

PROMPT_FOOTER = """
Wichtige Richtlinien:
- Halten Sie die Antworten prägnant und sachlich
- Verwenden Sie Deutsch für die Zusammenfassung
- Trennen Sie mehrere Elemente mit Semikolons (wo angegeben)
- Seien Sie objektiv in Ihrer Bewertung
- Falls Informationen unklar sind, geben Sie "unklar" an, anstatt zu raten
- Folgen Sie exakt dem angegebenen Format"""

def generate_analysis_prompt():
    """
    Dynamically generate the analysis prompt based on enabled fields.
    """
    prompt_parts = [PROMPT_HEADER]
    
    for field_key, field_config in ALL_ANALYSIS_FIELDS.items():
        if field_config.get("enabled", False):
            field_line = f"{field_config['prompt_name']}: [{field_config['description']}]"
            if "instructions" in field_config:
                field_line += f" - {field_config['instructions']}"
            prompt_parts.append(field_line)
    
    prompt_parts.append(PROMPT_FOOTER)
    return "\n".join(prompt_parts)

# Generate the actual prompt (this will be used by the analysis script)
ANALYSIS_PROMPT = generate_analysis_prompt()

# Prompt for summarizing long texts before analysis
SUMMARY_PROMPT_TEMPLATE = """
Erstellen Sie eine prägnante Zusammenfassung des folgenden deutschen Textes in maximal {max_words} Wörtern. 
Konzentrieren Sie sich auf das Hauptthema, die wichtigsten Punkte und die Gesamtbotschaft. Schreiben Sie auf Deutsch.

Text: {text}
"""

# =============================================================================
# CSV OUTPUT CONFIGURATION
# =============================================================================

def generate_field_mapping():
    """
    Dynamically generate field mapping based on enabled fields.
    """
    mapping = {}
    for field_key, field_config in ALL_ANALYSIS_FIELDS.items():
        if field_config.get("enabled", False):
            mapping[field_config["prompt_name"].lower()] = field_config["csv_column"]
    return mapping

def generate_list_fields():
    """
    Identify fields that contain lists based on field_type.
    """
    return [
        field_config["csv_column"] 
        for field_config in ALL_ANALYSIS_FIELDS.values() 
        if field_config.get("enabled", False) and field_config.get("field_type") == "list"
    ]

def generate_csv_columns():
    """
    Generate CSV column order based on enabled fields.
    """
    # Standard metadata columns (always included)
    standard_columns = [
        "file_name",
        "processed_at"
    ]
    
    # Analysis columns (based on enabled fields)
    analysis_columns = [
        field_config["csv_column"] 
        for field_config in ALL_ANALYSIS_FIELDS.values() 
        if field_config.get("enabled", False)
    ]
    
    # Technical metadata columns (always included)
    metadata_columns = [
        "file_size_chars",
        "tokens_used",
        "estimated_cost",
        "model"
    ]
    
    return standard_columns + analysis_columns + metadata_columns

# Generate dynamic configurations
FIELD_MAPPING = generate_field_mapping()
LIST_FIELDS = generate_list_fields()
CSV_COLUMNS = generate_csv_columns()

# Custom CSV column names (override defaults)
CUSTOM_CSV_COLUMNS = {
    # Example: change column names in CSV output
    # "main_topic": "Hauptthema_DE",
    # "sentiment": "Stimmung_DE",
    # "political_stance": "Politische_Ausrichtung"
}

# Apply custom column name overrides
for old_name, new_name in CUSTOM_CSV_COLUMNS.items():
    if old_name in CSV_COLUMNS:
        index = CSV_COLUMNS.index(old_name)
        CSV_COLUMNS[index] = new_name

# =============================================================================
# ANALYSIS PRESETS
# =============================================================================

# Quick analysis preset (fewer fields)
QUICK_ANALYSIS_PRESET = [
    "hauptthema", "stimmung", "zusammenfassung", "artikeltyp"
]

# Detailed analysis preset (all standard fields)
DETAILED_ANALYSIS_PRESET = [
    "hauptthema", "stimmung", "politische_haltung", "schluesselentitaeten", 
    "schluesselwoerter", "zusammenfassung", "verzerrungsgrad", 
    "glaubwuerdigkeit", "zielgruppe", "artikeltyp"
]

# Research analysis preset (includes custom fields)
RESEARCH_ANALYSIS_PRESET = [
    "hauptthema", "stimmung", "politische_haltung", "schluesselentitaeten",
    "schluesselwoerter", "zusammenfassung", "verzerrungsgrad", 
    "glaubwuerdigkeit", "zielgruppe", "artikeltyp", "emotionale_wirkung",
    "komplexitaet", "quellen_qualitaet"
]

def apply_analysis_preset(preset_name):
    """
    Apply a predefined analysis preset.
    
    Args:
        preset_name (str): Name of preset ("quick", "detailed", "research")
    """
    presets = {
        "quick": QUICK_ANALYSIS_PRESET,
        "detailed": DETAILED_ANALYSIS_PRESET, 
        "research": RESEARCH_ANALYSIS_PRESET
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    # Disable all fields first
    for field_key in ALL_ANALYSIS_FIELDS:
        ALL_ANALYSIS_FIELDS[field_key]["enabled"] = False
    
    # Enable only preset fields
    for field_key in presets[preset_name]:
        if field_key in ALL_ANALYSIS_FIELDS:
            ALL_ANALYSIS_FIELDS[field_key]["enabled"] = True

# Uncomment to apply a preset:
# apply_analysis_preset("detailed")

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# COST ESTIMATION
# =============================================================================

# OpenAI pricing per 1M tokens (as of 2024 - check current pricing)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
}

def get_estimated_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """
    Calculate estimated cost for GPT API usage.
    
    Args:
        prompt_tokens (int): Number of input tokens
        completion_tokens (int): Number of output tokens  
        model (str): Model name
        
    Returns:
        float: Estimated cost in USD
    """
    if model not in PRICING:
        # Default to gpt-4o-mini pricing if model not found
        model = "gpt-4o-mini"
    
    input_cost = (prompt_tokens / 1_000_000) * PRICING[model]["input"]
    output_cost = (completion_tokens / 1_000_000) * PRICING[model]["output"]
    
    return input_cost + output_cost

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """
    Validate configuration settings.
    
    Raises:
        ValueError: If configuration is invalid
    """
    errors = []
    
    # Check required fields
    if not ARTICLES_FOLDER:
        errors.append("ARTICLES_FOLDER cannot be empty")
    
    if not ANALYSIS_PROMPT:
        errors.append("ANALYSIS_PROMPT cannot be empty")
    
    if GPT_MAX_TOKENS < 100:
        errors.append("GPT_MAX_TOKENS must be at least 100")
    
    if MAX_CHUNK_SIZE < 1000:
        errors.append("MAX_CHUNK_SIZE must be at least 1000")
    
    if not FIELD_MAPPING:
        errors.append("FIELD_MAPPING cannot be empty")
    
    if not CSV_COLUMNS:
        errors.append("CSV_COLUMNS cannot be empty")
    
    # Check that at least one analysis field is enabled
    enabled_fields = [f for f in ALL_ANALYSIS_FIELDS.values() if f.get("enabled", False)]
    if not enabled_fields:
        errors.append("At least one analysis field must be enabled")
    
    # Check model exists in pricing
    if GPT_MODEL not in PRICING:
        print(f"Warning: Model '{GPT_MODEL}' not in pricing table, will use default pricing")
    
    # Check temperature range
    if not 0 <= GPT_TEMPERATURE <= 1:
        errors.append("GPT_TEMPERATURE must be between 0 and 1")
    
    if errors:
        raise ValueError("Configuration validation failed: " + "; ".join(errors))

# =============================================================================
# CUSTOM ANALYSIS PROMPTS (OPTIONAL)
# =============================================================================

# Specialized prompts for specific analysis types
SENTIMENT_ONLY_PROMPT = """
Analysieren Sie die Stimmung dieses deutschen Nachrichtenartikels. 
Antworten Sie nur mit einem Wort: positiv, negativ oder neutral.

Artikel: {text}
"""

BIAS_DETECTION_PROMPT = """
Analysieren Sie mögliche Verzerrungen in diesem deutschen Nachrichtenartikel.
Bewerten Sie den Verzerrungsgrad als: niedrig, mittel oder hoch
Erklären Sie Ihre Begründung in 2-3 Sätzen.

Verzerrungsgrad: [niedrig/mittel/hoch]
Begründung: [Erklärung]

Artikel: {text}
"""

ENTITY_EXTRACTION_PROMPT = """
Extrahieren Sie alle wichtigen benannten Entitäten aus diesem deutschen Nachrichtenartikel.
Einschließlich: Personen, Organisationen, Orte, Daten, Zahlen.

Format: Entitätstyp: Entitätsname; Entitätstyp: Entitätsname; ...

Artikel: {text}
"""

# =============================================================================
# ADVANCED SETTINGS (MODIFY WITH CARE)
# =============================================================================

# Text preprocessing options:
REMOVE_URLS = True  # Remove URLs from text before analysis
REMOVE_EMAILS = True  # Remove email addresses
NORMALIZE_WHITESPACE = True  # Normalize multiple spaces/newlines

# =============================================================================
# EXAMPLE USAGE CONFIGURATIONS
# =============================================================================

# Quick analysis (faster, cheaper):
QUICK_CONFIG = {
    "GPT_MODEL": "gpt-4o-mini",
    "GPT_MAX_TOKENS": 500,
    "ENABLE_STOPWORD_FILTERING": False,
    "ENABLE_SUMMARY": False,
    "DELAY_BETWEEN_FILES": 0.5
}

# Detailed analysis (slower, more expensive):
DETAILED_CONFIG = {
    "GPT_MODEL": "gpt-4o", 
    "GPT_MAX_TOKENS": 1500,
    "ENABLE_STOPWORD_FILTERING": True,
    "ENABLE_SUMMARY": True,
    "DELAY_BETWEEN_FILES": 2
}

# To use a preset configuration, uncomment these lines:
# config_preset = QUICK_CONFIG  # or DETAILED_CONFIG
# for key, value in config_preset.items():
#     globals()[key] = value

# =============================================================================
# CONFIGURATION EXAMPLES
# =============================================================================

"""
EXAMPLE 1: Add a custom analysis field
--------------------------------------
1. Add to CUSTOM_ANALYSIS_FIELDS:

"factual_accuracy": {
    "enabled": True,
    "prompt_name": "Faktische Genauigkeit",
    "description": "hoch/mittel/niedrig",
    "csv_column": "factual_accuracy",
    "field_type": "choice",
    "options": ["hoch", "mittel", "niedrig"],
    "instructions": "Bewerten Sie die Korrektheit der Fakten"
}

2. The field will automatically be included in prompts and CSV output!

EXAMPLE 2: Use only specific fields
-----------------------------------
1. Set specific fields to enabled=True, others to enabled=False
2. Or use apply_analysis_preset("quick") for predefined sets

EXAMPLE 3: Change CSV column names
----------------------------------
Add to CUSTOM_CSV_COLUMNS:
"main_topic": "Hauptthema_DE",
"sentiment": "Stimmung_DE"

EXAMPLE 4: Modify field descriptions
------------------------------------
Change the "description" in any field to customize the prompt text.
"""