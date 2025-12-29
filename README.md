# Multi-Agent Financial Research System

langgraph-based agentic ai system for financial analysis with prompt chaining, routing, and self-optimization. implements react-style agents with memory persistence for investment research queries.

built for AAI-520 group project, converted from colab notebook to modular local python for better deployment and sre practices.

## what it does

takes natural language queries about stocks and routes them to specialized handlers:
- **price**: current quotes, 52-week ranges, price targets
- **trend**: historical series, technical analysis, momentum
- **news**: headlines with nlp pipeline (sentiment + named entity recognition)
- **earnings**: quarterly reports, eps, guidance

the system uses:
- **prompt chaining**: fetch → preprocess → sentiment (vader) → ner (huggingface) → aggregate → summarize
- **intelligent routing**: regex rules with llm fallback for ambiguous queries
- **evaluator-optimizer**: scores answers 0-4, reruns with adjustments if below threshold
- **memory persistence**: remembers context across conversation turns via thread_id

## architecture

```
financial_agent/
├── data/                   # cache and storage
├── logs/                   # analysis outputs
├── models/                 # saved model artifacts
├── tools/                  # custom tool extensions
├── config.py              # env loader and api key management
├── data_fetching.py       # price/news from finnhub/alphavantage/yfinance
├── nlp_pipeline.py        # vader sentiment + hf ner + aggregation
├── routing.py             # intent classification with regex + llm
├── evaluator.py           # answer quality scoring and optimization
├── tools.py               # langchain tool wrappers
├── agent.py               # langgraph agent initialization
├── main.py                # interactive demo
├── analyze.py             # routing metrics and quality analysis
└── requirements.txt
```

## setup

1. clone and install dependencies
```bash
git clone <repo-url>
cd financial_agent
pip install -r requirements.txt
```

2. get api keys from:
   - [tavily](https://tavily.com) - web search
   - [xai](https://x.ai/api) - grok-2 llm
   - [finnhub](https://finnhub.io) - financial data (optional)
   - [alpha vantage](https://www.alphavantage.co) - market data (optional)
   - [langsmith](https://smith.langchain.com) - tracing (optional)

3. configure environment
```bash
cp .env.example .env
# edit .env with your keys
```

4. run demo
```bash
python main.py
```

## usage

**interactive demo:**
```bash
python main.py
```

runs predefined queries showing routing, tool usage, and memory persistence.

**analysis and metrics:**
```bash
python analyze.py
```

generates routing coverage stats, quality scores, and visualizations.

**custom queries:**
```python
from config import load_config
from agent import init_agent, ask_agent, get_answer

config = load_config()
agent, _ = init_agent(config)

result = ask_agent(agent, "analyze recent NVDA news", thread_id="my-session")
print(get_answer(result))
```

## key features

### prompt chaining pipeline
news analysis goes through full nlp workflow:
1. **fetch** - get articles from finnhub/yfinance
2. **preprocess** - clean text, remove urls
3. **sentiment** - vader polarity scores (pos/neu/neg)
4. **ner** - extract entities with huggingface transformers
5. **aggregate** - sentiment percentages, top entities
6. **summarize** - format human-readable report

### intelligent routing
queries classified via:
- **deterministic rules** (regex patterns) - fast path for common queries
- **llm fallback** (grok-2) - handles ambiguous or complex requests
- **debug mode** - shows which rule matched for troubleshooting

### evaluator-optimizer loop
answers scored on 0-4 scale:
- 0 = irrelevant/wrong
- 1 = partially relevant but incomplete
- 2 = relevant but missing details
- 3 = good with minor gaps
- 4 = comprehensive and accurate

if score < 3, system can:
- regenerate answer with more context
- increase article limit for news analysis
- add clarifying details

### memory persistence
uses langgraph's memorysaver with thread_id:
- remembers user preferences across turns
- maintains conversation context
- separate threads for different sessions

## data sources

**price data:**
- primary: finnhub api (best for real-time quotes)
- fallback: yfinance (free but slower)

**historical series:**
- primary: alpha vantage (better rate limits)
- fallback: yfinance

**news:**
- primary: finnhub company news (includes summaries)
- fallback: yfinance news feed

**nlp models:**
- sentiment: nltk vader lexicon (fast, finance-tuned)
- ner: dslim/bert-base-ner (small footprint, good accuracy)

## notes and gotchas

- **api rate limits**: finnhub/alphavantage have free tier limits. yfinance is unlimited but slower
- **langsmith tracing**: set ENABLE_TRACING=false if you don't have a langsmith account
- **memory persistence**: thread_id must be consistent across calls to maintain context
- **routing ambiguity**: some queries match multiple patterns. llm fallback handles these but costs tokens
- **vader limitations**: trained on general text, not specifically financial news. still works reasonably well for sentiment trends
- **ner entity types**: focuses on PERSON, ORG, LOC. product names and technical terms may be missed

## performance characteristics

**routing coverage** (on test set):
- price: ~25%
- trend: ~20%
- news: ~35%
- earnings: ~20%

**quality scores** (baseline):
- avg score: 2.8/4.0 before optimization
- avg score: 3.4/4.0 after optimization

**latency** (approximate):
- price query: ~1s (api call)
- trend query: ~2s (api + data processing)
- news analysis: ~8-12s (fetch + nlp pipeline)
- with llm routing: +1-2s per query

## team contributions

- callum lamb: team lead, agent architecture
- jack baxter: prompt chaining, nlp pipeline
- deepti pamula: routing implementation
- jasper dolar: evaluator-optimizer

## acknowledgments

- langchain/langgraph for agent framework
- xai for grok-2 llm access
- finnhub, alpha vantage, yfinance for financial data
- huggingface for ner models
- nltk vader for sentiment analysis

ai assistance from grok (xai) and chatgpt (openai) for code acceleration - all outputs reviewed and validated by team.

## license

mit - academic project
