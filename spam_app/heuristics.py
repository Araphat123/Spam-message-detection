import re

def calculate_heuristic_score(text):
    """
    Calculates a 'spamminess' score based on rules and keywords.
    Returns a float between 0.0 (looks safe) and 1.0 (definitely spam).
    """
    score = 0.0
    text_lower = text.lower()
    
    # 1. KEYWORDS (Urgency, Money, Action)
    # Weighted by severity
    keywords = {
        'urgent': 0.3, 'immediately': 0.2, 'act now': 0.3, 'verify': 0.2,
        'winner': 0.4, 'won': 0.3, 'prize': 0.3, 'lottery': 0.4,
        'cash': 0.2, 'money': 0.2, 'investment': 0.2, 'crypto': 0.3,
        'bitcoin': 0.3, 'bank account': 0.3, 'suspended': 0.3,
        'unusual activity': 0.2, 'confirm your': 0.2, 'free': 0.2,
        'offer': 0.1, 'deal': 0.1, 'click here': 0.3, 'subscribe': 0.1
    }
    
    for word, weight in keywords.items():
        if word in text_lower:
            score += weight
            
    # 2. PATTERNS
    
    # URL presence (High risk)
    if re.search(r'https?://\S+', text) or re.search(r'www\.\S+', text):
        score += 0.4
        
    # Phone numbers (e.g. +1-234...) often used in spam
    if re.search(r'\+?\d[\d -]{8,12}\d', text):
        score += 0.2
        
    # Excessive Caps (SHOUTING)
    caps_count = sum(1 for c in text if c.isupper())
    if len(text) > 10 and caps_count / len(text) > 0.4:
        score += 0.3
        
    # Dollar signs
    if '$' in text:
        score += 0.2
        
    # 3. SAFE KEYWORDS (Context clues for Ham)
    # These reduce the score to prevent false positives
    safe_keywords = {
        'mom': 0.2, 'dad': 0.2, 'home': 0.1, 'meeting': 0.2, 'assignment': 0.2,
        'class': 0.1, 'project': 0.1, 'love': 0.1, 'sorry': 0.1, 'later': 0.1,
        'ok': 0.1, 'okay': 0.1, 'thanks': 0.1, 'hey': 0.05, 'today': 0.05,
        'tomorrow': 0.05, 'work': 0.1, 'report': 0.1, 'presentation': 0.1
    }
    
    for word, reduction in safe_keywords.items():
        if word in text_lower:
            score -= reduction
            
    # 4. SAFE PHRASES (Context is key)
    # These handle multi-word common safe expressions
    safe_phrases = {
        'are you free': 0.5,           # Contextual "free" (time) vs "free" (money)
        'when you are free': 0.5,
        'can you call': 0.3,
        'call me when': 0.3,
        'let me know': 0.2,
        'how are you': 0.2,
        'see you': 0.2
    }
    
    for phrase, reduction in safe_phrases.items():
        if phrase in text_lower:
            score -= reduction
            
    # 3. CAP SCORE
    # We cap the heuristic contribution so it doesn't override ML completely unless overwhelming
    # Also ensure it doesn't go below 0
    return max(0.0, min(score, 1.0))
