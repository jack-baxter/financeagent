#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

from config import load_config, validate_config, print_config
from agent import init_agent, ask_agent, get_answer

def main():
    #load config and validate api keys
    config = load_config()
    print_config(config)
    
    if not validate_config(config):
        print("\nerror: missing required api keys")
        print("copy .env.example to .env and add your keys")
        return
    
    #initialize agent with memory
    print("\ninitializing agent...")
    agent, memory = init_agent(config)
    print("agent ready")
    
    #demo queries to test routing and tool usage
    test_queries = [
        "what's the current price for NVDA?",
        "show me the last 30 days trend for AAPL",
        "analyze recent news for MSFT with sentiment",
        "give me quick headlines for TSLA"
    ]
    
    thread_id = "demo-session"
    
    print("\n" + "="*60)
    print("running demo queries")
    print("="*60)
    
    for query in test_queries:
        print(f"\nquery: {query}")
        print("-"*60)
        
        result = ask_agent(agent, query, thread_id=thread_id)
        answer = get_answer(result)
        
        print(answer)
        print()
    
    #memory persistence demo
    print("\n" + "="*60)
    print("memory persistence test")
    print("="*60)
    
    #first turn
    result1 = ask_agent(agent, "remember that i'm interested in NVDA", thread_id="memory-test")
    print("\nturn 1:", get_answer(result1))
    
    #second turn on same thread
    result2 = ask_agent(agent, "what was the stock i mentioned?", thread_id="memory-test")
    print("\nturn 2:", get_answer(result2))

if __name__ == "__main__":
    main()
