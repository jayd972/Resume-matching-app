import re

def extract_skills(text):
    # Simple approach: find "Skills" or "Technical Skills" section; fallback to keywords
    skills_regex = r'(Skills|Technical Skills)[:\-\n]+([\s\S]+?)(?:\n[A-Z][A-Za-z ]+:|\Z)'
    match = re.search(skills_regex, text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    # Fallback: extract comma/line separated words likely to be skills
    lines = text.splitlines()
    for line in lines:
        if 'skill' in line.lower():
            return line
    return ""

def extract_experience(text):
    # Look for "Experience" or "Professional Experience" section
    exp_regex = r'(Experience|Professional Experience)[:\-\n]+([\s\S]+?)(?:\n[A-Z][A-Za-z ]+:|\Z)'
    match = re.search(exp_regex, text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    # Fallback: return first big block if no explicit section
    return "\n".join(text.splitlines()[0:15])
