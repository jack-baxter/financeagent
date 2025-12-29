from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from tools import get_all_tools
from routing import set_routing_llm
from evaluator import set_eval_llm
from data_fetching import init_vendors
from nlp_pipeline import init_nlp_models

#system prompt defines agent behavior and tool usage
SYSTEM_PROMPT = """you are an investment research assistant. use tools when helpful.

available tools:
- headlines/summaries → get_recent_news / analyze_news_report
- current price → get_latest_price
- historical trend → get_daily_series

cite publishers by name and keep answers concise.
for news analysis, use analyze_news_report for detailed sentiment/entity extraction.
for quick headlines, use get_recent_news."""

#initialize agent with memory and tools
#returns agent instance and memory checkpointer
def init_agent(config: dict):
    #init llm
    llm = init_chat_model(
        model=config['model_name'],
        model_provider="xai",
        temperature=config['model_temperature']
    )
    
    #init vendor clients and nlp models
    init_vendors()
    init_nlp_models(config['ner_model'])
    
    #set llm for routing and evaluation
    set_routing_llm(llm)
    set_eval_llm(llm)
    
    #create memory checkpointer
    memory = MemorySaver()
    
    #create react agent with tools
    tools = get_all_tools()
    agent = create_react_agent(llm, tools, checkpointer=memory)
    
    return agent, memory

#invoke agent with memory thread support
def ask_agent(agent, message: str, thread_id: str = "default", 
              system_prompt: str = SYSTEM_PROMPT) -> dict:
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(system_prompt),
                ("user", message)
            ]
        },
        config={"configurable": {"thread_id": thread_id}}
    )
    
    return result

#extract final answer from agent result
def get_answer(result: dict) -> str:
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "no response"

#batch processing with routing
def batch_process(agent, prompts: list, symbol: str = "NVDA", 
                 thread_id: str = "batch") -> list:
    from routing import route_intent
    
    results = []
    for prompt in prompts:
        bucket = route_intent(prompt)
        result = ask_agent(agent, prompt, thread_id)
        answer = get_answer(result)
        
        results.append({
            'prompt': prompt,
            'bucket': bucket,
            'answer': answer
        })
    
    return results
