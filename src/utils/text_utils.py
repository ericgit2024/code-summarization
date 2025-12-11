import re

def extract_overview(summary_text):
    """
    Extracts the 'Overview' section from the structured markdown summary.
    Falls back to the full text if no headers are found.
    """
    if not summary_text:
        return ""

    # Pattern to find the Overview section
    # Matches "### Overview" or "**Overview**" or "1. **Overview**"
    # Captures everything until the next header (### or **Section**)

    # Normalize headers to ### for easier parsing if they vary
    text = summary_text.replace("**Overview**", "### Overview")
    text = re.sub(r'^\d+\.\s+###', '###', text, flags=re.MULTILINE) # Handle "1. ### Overview"

    match = re.search(r'### Overview\s*(.*?)\s*(?=###|$)', text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # If no overview section explicitly found, return the whole text
    # (Maybe it's a short summary already)
    return summary_text.strip()

def nlp_to_codexglue(summary):
    """
    Converts a generated summary (possibly with Markdown headers) into a plain text format
    comparable to CodeXGLUE reference docstrings.

    This essentially wraps extract_overview but adds additional cleaning steps
    to ensure the output is suitable for NLP metric evaluation (single line, no markdown artifacts).
    """
    if not summary:
        return ""

    # 1. Extract the overview section
    text = extract_overview(summary)

    # 2. Additional cleaning
    # Remove markdown bold/italic
    text = text.replace('**', '').replace('*', '')

    # Remove code backticks
    text = text.replace('`', '')

    # Normalize whitespace (replace newlines with spaces)
    text = " ".join(text.split())

    return text
