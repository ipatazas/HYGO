import re
from collections import OrderedDict

def parse_parameter_file(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Clean LaTeX-like syntax
    cleaned = (
        text.replace('\\texttt{', '')
            .replace('\\textit{', '')
            .replace('\\textbf{', '')
            .replace('\\_', '_')
            .replace('\\HYGO', 'HYGO')
            .replace('{', '')
            .replace('}', '')
    )

    # Regex pattern to extract key-value pairs
    pattern = r'item\s+([a-zA-Z0-9_]+)\s+\[[^\]]*\]\s*:\s*(.*?)\s*(?=item\s+[a-zA-Z0-9_]+\s+\[|$)'
    matches = re.findall(pattern, cleaned, re.DOTALL)

    # Build ordered dictionary
    param_dict = OrderedDict((key.strip(), value.strip()) for key, value in matches)
    return param_dict