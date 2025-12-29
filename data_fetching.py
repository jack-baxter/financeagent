from datetime import datetime, timedelta, date
import os
import yfinance as yf

#lazy load vendor clients to avoid import errors if keys missing
_finnhub_client = None
_av_ts = None

def init_vendors():
    global _finnhub_client, _av_ts
    
    #finnhub client
    try:
        import finnhub
        if os.getenv("FINNHUB_API_KEY"):
            _finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
    except Exception:
        _finnhub_client = None
    
    #alpha vantage time series
    try:
        from alpha_vantage.timeseries import TimeSeries
        if os.getenv("ALPHAVANTAGE_API_KEY"):
            _av_ts = TimeSeries(key=os.getenv("ALPHAVANTAGE_API_KEY"), output_format='pandas')
    except Exception:
        _av_ts = None

#get latest price for ticker, tries vendors then yfinance fallback
def get_latest_price(symbol: str) -> dict:
    symbol = symbol.upper().strip()
    
    #try finnhub first
    if _finnhub_client:
        try:
            quote = _finnhub_client.quote(symbol)
            if quote and quote.get('c'):
                return {
                    'symbol': symbol,
                    'price': quote['c'],
                    'change': quote.get('d', 0),
                    'percent_change': quote.get('dp', 0),
                    'source': 'finnhub'
                }
        except Exception as e:
            print(f"finnhub error for {symbol}: {e}")
    
    #fallback to yfinance
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if price:
            return {
                'symbol': symbol,
                'price': price,
                'change': info.get('regularMarketChange', 0),
                'percent_change': info.get('regularMarketChangePercent', 0),
                'source': 'yfinance'
            }
    except Exception as e:
        print(f"yfinance error for {symbol}: {e}")
    
    return {'symbol': symbol, 'error': 'could not fetch price'}

#get daily price series for trend analysis
def get_daily_series(symbol: str, days: int = 30) -> dict:
    symbol = symbol.upper().strip()
    days = max(1, min(int(days), 365))
    
    #try alpha vantage first for better rate limits
    if _av_ts:
        try:
            data, _ = _av_ts.get_daily(symbol=symbol, outputsize='compact')
            recent = data.tail(days)
            
            return {
                'symbol': symbol,
                'days': len(recent),
                'data': recent.to_dict('records'),
                'source': 'alphavantage'
            }
        except Exception as e:
            print(f"alphavantage error for {symbol}: {e}")
    
    #fallback to yfinance
    try:
        ticker = yf.Ticker(symbol)
        end = datetime.now()
        start = end - timedelta(days=days)
        hist = ticker.history(start=start, end=end)
        
        if not hist.empty:
            return {
                'symbol': symbol,
                'days': len(hist),
                'data': hist.to_dict('records'),
                'source': 'yfinance'
            }
    except Exception as e:
        print(f"yfinance error for {symbol}: {e}")
    
    return {'symbol': symbol, 'error': 'could not fetch series'}

#fetch recent news articles, prefers finnhub for better summaries
def fetch_news(symbol: str, limit: int = 12) -> list:
    symbol = symbol.upper().strip()
    limit = max(1, min(int(limit), 25))
    
    items = []
    
    #try finnhub company news first
    if _finnhub_client:
        try:
            end = date.today()
            start = end - timedelta(days=7)
            news = _finnhub_client.company_news(
                symbol, 
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d")
            )
            
            for item in (news or [])[:limit]:
                items.append({
                    'title': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'publisher': item.get('source', ''),
                    'link': item.get('url', ''),
                    'published': item.get('datetime', 0)
                })
            
            if items:
                return items
        except Exception as e:
            print(f"finnhub news error for {symbol}: {e}")
    
    #fallback to yfinance
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        
        for item in news[:limit]:
            items.append({
                'title': item.get('title', ''),
                'summary': '',
                'publisher': item.get('publisher', ''),
                'link': item.get('link', ''),
                'published': item.get('providerPublishTime', 0)
            })
    except Exception as e:
        print(f"yfinance news error for {symbol}: {e}")
    
    return items
