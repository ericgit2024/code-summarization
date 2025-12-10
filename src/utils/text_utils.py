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
