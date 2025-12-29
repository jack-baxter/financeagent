from langchain_core.tools import tool
from data_fetching import get_latest_price, get_daily_series, fetch_news
from nlp_pipeline import analyze_news
from evaluator import evaluate_news_analysis

#wrapper tool for getting current stock price
@tool("get_latest_price")
def get_price_tool(symbol: str) -> str:
    result = get_latest_price(symbol)
    
    if 'error' in result:
        return f"error fetching price for {symbol}: {result['error']}"
    
    return (f"{result['symbol']}: ${result['price']:.2f} "
            f"({result['change']:+.2f}, {result['percent_change']:+.2f}%) "
            f"[source: {result['source']}]")

#wrapper tool for getting historical price series
@tool("get_daily_series")
def get_series_tool(symbol: str, days: int = 30) -> str:
    result = get_daily_series(symbol, days)
    
    if 'error' in result:
        return f"error fetching series for {symbol}: {result['error']}"
    
    data = result.get('data', [])
    if not data:
        return f"no price data available for {symbol}"
    
    recent = data[-5:] if len(data) >= 5 else data
    summary = "\n".join([
        f"  {d.get('date', 'n/a')}: ${d.get('close', 0):.2f}"
        for d in recent
    ])
    
    return (f"{result['symbol']} - last {result['days']} days:\n{summary}\n"
            f"[source: {result['source']}]")

#wrapper tool for news analysis with nlp pipeline
@tool("analyze_news_report")
def analyze_news_tool(symbol: str, limit: int = 12, bullets: int = 4) -> str:
    #fetch news
    news_items = fetch_news(symbol, limit)
    
    if not news_items:
        return f"no recent news found for {symbol}"
    
    #run through nlp pipeline
    analysis = analyze_news(symbol, news_items)
    report = analysis['report']
    
    #evaluate quality
    evaluation = evaluate_news_analysis(analysis)
    
    #format summary
    sentiment = report['sentiment_pct']
    entities = list(report['top_entities'].items())[:5]
    
    summary = f"{symbol} news analysis ({report['n_articles']} articles):\n\n"
    summary += f"sentiment: {sentiment['positive']:.1f}% positive, "
    summary += f"{sentiment['neutral']:.1f}% neutral, "
    summary += f"{sentiment['negative']:.1f}% negative\n\n"
    
    if entities:
        summary += "key entities: " + ", ".join([f"{e[0]} ({e[1]})" for e in entities]) + "\n\n"
    
    #top headlines as bullets
    summary += "recent headlines:\n"
    for i, item in enumerate(analysis['items'][:bullets], 1):
        pub = item.get('publisher', 'unknown')
        title = item.get('title', 'no title')
        sent = item['sentiment']['label']
        summary += f"{i}. [{pub}] {title} ({sent})\n"
    
    summary += f"\n[quality score: {evaluation['score']}/4]"
    
    return summary

#simple tool for getting recent headlines without full analysis
@tool("get_recent_news")
def get_news_tool(symbol: str, limit: int = 5) -> str:
    news_items = fetch_news(symbol, limit)
    
    if not news_items:
        return f"no recent news found for {symbol}"
    
    headlines = []
    for i, item in enumerate(news_items, 1):
        pub = item.get('publisher', 'unknown')
        title = item.get('title', 'no title')
        headlines.append(f"{i}. [{pub}] {title}")
    
    return f"{symbol} recent news:\n" + "\n".join(headlines)

#get all tools as list for agent initialization
def get_all_tools():
    return [
        get_price_tool,
        get_series_tool,
        analyze_news_tool,
        get_news_tool
    ]
