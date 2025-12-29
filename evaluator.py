from typing import Optional

#llm instance for evaluation, set by agent init
_eval_llm = None

def set_eval_llm(llm):
    global _eval_llm
    _eval_llm = llm

#evaluate answer quality on 0-4 scale
#checks completeness, accuracy, relevance
def evaluate_answer(query: str, answer: str, context: Optional[dict] = None) -> dict:
    if not _eval_llm:
        return {'score': None, 'feedback': 'evaluator llm not initialized'}
    
    try:
        prompt = f"""evaluate this financial research answer on a 0-4 scale:

query: {query}

answer: {answer}

scoring:
0 - irrelevant or wrong
1 - partially relevant but incomplete
2 - relevant but missing key details
3 - good answer with minor gaps
4 - comprehensive and accurate

respond with only the score (0-4) followed by brief feedback.
format: SCORE: X
FEEDBACK: your feedback here"""
        
        response = _eval_llm.invoke(prompt)
        content = response.content.strip()
        
        #parse score
        score_line = [l for l in content.split('\n') if l.startswith('SCORE:')]
        score = None
        if score_line:
            try:
                score = int(score_line[0].split(':')[1].strip())
            except:
                pass
        
        #parse feedback
        feedback_lines = [l for l in content.split('\n') if l.startswith('FEEDBACK:')]
        feedback = feedback_lines[0].split(':', 1)[1].strip() if feedback_lines else content
        
        return {
            'score': score,
            'feedback': feedback
        }
    except Exception as e:
        return {'score': None, 'feedback': f'evaluation error: {e}'}

#optimize answer based on evaluation feedback
#generates improved version if score below threshold
def optimize_answer(query: str, answer: str, evaluation: dict, 
                   context: Optional[dict] = None) -> dict:
    score = evaluation.get('score')
    
    if score is None or score >= 3:
        return {
            'optimized': False,
            'answer': answer,
            'reason': 'score acceptable or evaluation failed'
        }
    
    if not _eval_llm:
        return {
            'optimized': False,
            'answer': answer,
            'reason': 'optimizer llm not initialized'
        }
    
    try:
        prompt = f"""improve this financial research answer based on feedback:

query: {query}

current answer: {answer}

feedback: {evaluation.get('feedback', 'incomplete')}

provide an improved version that addresses the feedback.
keep it concise and factual."""
        
        response = _eval_llm.invoke(prompt)
        improved = response.content.strip()
        
        return {
            'optimized': True,
            'answer': improved,
            'reason': f"improved from score {score}"
        }
    except Exception as e:
        return {
            'optimized': False,
            'answer': answer,
            'reason': f'optimization error: {e}'
        }

#evaluate news analysis quality based on data completeness
#checks article count, sentiment distribution, entity extraction
def evaluate_news_analysis(analysis: dict, min_articles: int = 5) -> dict:
    report = analysis.get('report', {})
    n_articles = report.get('n_articles', 0)
    sentiment_pct = report.get('sentiment_pct', {})
    entities = report.get('top_entities', {})
    
    issues = []
    
    #check article coverage
    if n_articles < min_articles:
        issues.append(f"only {n_articles} articles (target: {min_articles}+)")
    
    #check sentiment distribution
    if all(v == 0 for v in sentiment_pct.values()):
        issues.append("no sentiment data")
    
    #check entity extraction
    if not entities:
        issues.append("no entities extracted")
    
    #score based on completeness
    if not issues:
        score = 4
    elif len(issues) == 1:
        score = 3
    elif len(issues) == 2:
        score = 2
    else:
        score = 1
    
    return {
        'score': score,
        'n_articles': n_articles,
        'issues': issues,
        'feedback': '; '.join(issues) if issues else 'analysis looks good'
    }

#optimize news analysis by fetching more articles if needed
def optimize_news_analysis(symbol: str, analysis: dict, 
                          min_score: int = 3, max_retries: int = 1) -> dict:
    evaluation = evaluate_news_analysis(analysis)
    
    if evaluation['score'] >= min_score:
        return {
            'analysis': analysis,
            'evaluation': evaluation,
            'optimized': False
        }
    
    #if score low, suggest increasing article limit
    current_articles = evaluation['n_articles']
    suggested_limit = min(current_articles * 2, 25)
    
    return {
        'analysis': analysis,
        'evaluation': evaluation,
        'optimized': False,
        'suggestion': f"increase news limit to {suggested_limit} for better coverage"
    }
