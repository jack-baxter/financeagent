import re
from collections import Counter
from typing import Optional

#lazy load nlp models to avoid slow imports on module load
_vader = None
_hf_ner = None

def init_nlp_models(ner_model: str = "dslim/bert-base-NER"):
    global _vader, _hf_ner
    
    #vader sentiment analyzer
    if _vader is None:
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    
    #huggingface ner pipeline
    if _hf_ner is None:
        from transformers import pipeline
        _hf_ner = pipeline("ner", model=ner_model, aggregation_strategy="simple")

#clean text for sentiment analysis
#removes urls, extra whitespace, normalizes case
def preprocess_text(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#run vader sentiment on text
#returns dict with pos/neu/neg scores and compound
def get_sentiment(text: str) -> dict:
    if not _vader:
        init_nlp_models()
    
    clean = preprocess_text(text)
    scores = _vader.polarity_scores(clean)
    
    #categorize based on compound score
    if scores['compound'] >= 0.05:
        label = 'positive'
    elif scores['compound'] <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'label': label,
        'compound': scores['compound'],
        'pos': scores['pos'],
        'neu': scores['neu'],
        'neg': scores['neg']
    }

#extract named entities using hf transformers
#returns list of entity dicts with text and type
def extract_entities(text: str, min_score: float = 0.5) -> list:
    if not _hf_ner:
        init_nlp_models()
    
    clean = preprocess_text(text)
    if not clean:
        return []
    
    try:
        entities = _hf_ner(clean)
        return [
            {'text': ent['word'], 'type': ent['entity_group'], 'score': ent['score']}
            for ent in entities if ent['score'] >= min_score
        ]
    except Exception as e:
        print(f"ner error: {e}")
        return []

#process single news item through sentiment and ner pipeline
def process_news_item(item: dict) -> dict:
    text = f"{item.get('title', '')} {item.get('summary', '')}"
    
    sentiment = get_sentiment(text)
    entities = extract_entities(text)
    
    return {
        'title': item.get('title', ''),
        'summary': item.get('summary', ''),
        'publisher': item.get('publisher', ''),
        'link': item.get('link', ''),
        'published': item.get('published', 0),
        'sentiment': sentiment,
        'entities': entities
    }

#aggregate sentiment percentages from processed items
def aggregate_sentiment(items: list) -> dict:
    if not items:
        return {'positive': 0, 'neutral': 0, 'negative': 0}
    
    counts = Counter(item['sentiment']['label'] for item in items)
    total = len(items)
    
    return {
        'positive': round(100 * counts.get('positive', 0) / total, 1),
        'neutral': round(100 * counts.get('neutral', 0) / total, 1),
        'negative': round(100 * counts.get('negative', 0) / total, 1)
    }

#aggregate top entities across all items
def aggregate_entities(items: list, top_k: int = 10) -> dict:
    all_entities = []
    for item in items:
        all_entities.extend([e['text'] for e in item.get('entities', [])])
    
    counts = Counter(all_entities)
    return dict(counts.most_common(top_k))

#full news analysis pipeline orchestration
#fetch -> preprocess -> sentiment -> ner -> aggregate
def analyze_news(symbol: str, news_items: list) -> dict:
    if not news_items:
        return {
            'symbol': symbol,
            'items': [],
            'report': {
                'n_articles': 0,
                'sentiment_pct': {'positive': 0, 'neutral': 0, 'negative': 0},
                'top_entities': {}
            }
        }
    
    #process each item through pipeline
    processed = [process_news_item(item) for item in news_items]
    
    #aggregate metrics
    sentiment_pct = aggregate_sentiment(processed)
    top_entities = aggregate_entities(processed)
    
    return {
        'symbol': symbol,
        'items': processed,
        'report': {
            'n_articles': len(processed),
            'sentiment_pct': sentiment_pct,
            'top_entities': top_entities
        }
    }
