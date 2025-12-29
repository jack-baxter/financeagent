import os
from dotenv import load_dotenv

#load config from env file and set os env vars for langchain/tavily/etc
def load_config() -> dict:
    load_dotenv()
    
    #api keys
    config = {
        'tavily_api_key': os.getenv('TAVILY_API_KEY', ''),
        'xai_api_key': os.getenv('XAI_API_KEY', ''),
        'langsmith_api_key': os.getenv('LANGSMITH_API_KEY', ''),
        'finnhub_api_key': os.getenv('FINNHUB_API_KEY', ''),
        'alphavantage_api_key': os.getenv('ALPHAVANTAGE_API_KEY', ''),
        'langsmith_project': os.getenv('LANGSMITH_PROJECT', 'financial-agent'),
        'enable_tracing': os.getenv('ENABLE_TRACING', 'true').lower() == 'true',
        'model_name': os.getenv('MODEL_NAME', 'grok-2-1212'),
        'model_temperature': float(os.getenv('MODEL_TEMPERATURE', 0)),
        'default_news_limit': int(os.getenv('DEFAULT_NEWS_LIMIT', 12)),
        'default_bullets': int(os.getenv('DEFAULT_BULLETS', 4)),
        'ner_model': os.getenv('NER_MODEL', 'dslim/bert-base-NER'),
        'min_score_threshold': int(os.getenv('MIN_SCORE_THRESHOLD', 3)),
        'refinement_retry_limit': int(os.getenv('REFINEMENT_RETRY_LIMIT', 20))
    }
    
    #set env vars that langchain expects
    os.environ['TAVILY_API_KEY'] = config['tavily_api_key']
    os.environ['XAI_API_KEY'] = config['xai_api_key']
    os.environ['LANGSMITH_API_KEY'] = config['langsmith_api_key']
    os.environ['LANGSMITH_PROJECT'] = config['langsmith_project']
    os.environ['FINNHUB_API_KEY'] = config['finnhub_api_key']
    os.environ['ALPHAVANTAGE_API_KEY'] = config['alphavantage_api_key']
    
    if config['enable_tracing']:
        os.environ['LANGSMITH_TRACING'] = 'true'
    
    return config

#validate required api keys present
def validate_config(config: dict) -> bool:
    required = ['tavily_api_key', 'xai_api_key']
    missing = [k for k in required if not config.get(k)]
    
    if missing:
        print(f"error: missing required api keys: {', '.join(missing)}")
        return False
    
    return True

#print sanitized config for debugging without exposing keys
def print_config(config: dict):
    print("current configuration:")
    print("-" * 50)
    for key, value in config.items():
        if 'key' in key.lower():
            display = f"{value[:8]}..." if value else "(not set)"
        else:
            display = value
        print(f"{key}: {display}")
    print("-" * 50)
