import re
from typing import Literal, Optional

Route = Literal["news", "trend", "price", "earnings"]

#deterministic regex patterns for common query types
#checked before llm fallback to save tokens and latency
ROUTE_RULES = {
    "price": re.compile(r"\b(price|quote|intraday|open|close|52[-\s]?week|target)\b", re.I),
    "earnings": re.compile(r"\b(earnings|eps|guidance|profit|revenue|quarterly)\b", re.I),
    "news": re.compile(r"\b(news|headline|article|story|catalyst|announcement)\b", re.I),
    "trend": re.compile(r"\b(trend|chart|technical|moving average|momentum|series)\b", re.I),
}

#llm instance for fallback routing, set by agent init
_llm = None

def set_routing_llm(llm):
    global _llm
    _llm = llm

#route user intent to appropriate specialist
#tries regex first, falls back to llm if no match
def route_intent(msg: str, use_llm: bool = True) -> Route:
    text = str(msg)
    
    #check deterministic rules first
    for route_name, pattern in ROUTE_RULES.items():
        if pattern.search(text):
            return route_name
    
    #llm fallback for ambiguous queries
    if use_llm and _llm:
        try:
            prompt = f"""classify this user query into one category:
- price: current price, quote, 52-week high/low
- earnings: earnings reports, eps, guidance
- news: headlines, articles, recent announcements
- trend: historical price movement, charts, technical analysis

query: {text}

respond with only the category name (price/earnings/news/trend)"""
            
            response = _llm.invoke(prompt)
            label = response.content.strip().lower()
            
            if label in ROUTE_RULES:
                return label
        except Exception as e:
            print(f"llm routing error: {e}")
    
    #default fallback
    return "news"

#debug wrapper to show which rule matched
def debug_route(msg: str) -> tuple[Route, str]:
    text = str(msg)
    
    #check regex
    for route_name, pattern in ROUTE_RULES.items():
        if pattern.search(text):
            return route_name, f"regex: {pattern.pattern[:50]}"
    
    #llm fallback
    route = route_intent(text)
    source = "llm" if _llm else "default"
    return route, source

#get routing statistics for debugging
def routing_stats(prompts: list) -> dict:
    from collections import Counter
    
    routes = [route_intent(p) for p in prompts]
    counts = Counter(routes)
    
    return {
        'total': len(prompts),
        'distribution': dict(counts),
        'coverage': {k: round(100 * v / len(prompts), 1) for k, v in counts.items()}
    }
