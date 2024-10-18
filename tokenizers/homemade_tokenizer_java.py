import re

def remove_comments_java(code):
    """Removes both single-line and multi-line comments from Java code."""
    comment_pattern = r'//.*?$|/\*.*?\*/'
    no_comments_code = re.sub(comment_pattern, '', code, flags=re.DOTALL | re.MULTILINE)
    return no_comments_code

def tokenize_java_code(code):
    # List of multi-character operators that should not be split
    multi_char_operators = ['==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '&&', '||', '++', '--', '::', '?.']

    # Regex pattern to match variables, numbers, multi-character operators, and symbols
    pattern = r'(\w+|==|!=|<=|>=|\+=|-=|\*=|/=|&&|\|\||\+\+|--|::|\?.|[+\-*/=(){};<>.,:])'
    tokens = []
    for line in code.splitlines():
        line_tokens = re.findall(pattern, line.strip())
        tokens.extend(line_tokens)
    return tokens

def is_java_keyword(token):
    java_keywords = [
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue',
        'default', 'do', 'double', 'else', 'enum', 'extends', 'false', 'final', 'finally', 'float', 'for', 'goto',
        'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'null', 'package',
        'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized',
        'this', 'throw', 'throws', 'transient', 'true', 'try', 'void', 'volatile', 'while'
    ]
    return token in java_keywords

def normalize_identifiers_java(tokens):
    var_count = 1
    fn_count = 1
    variables = {}
    functions = {}

    normalized_tokens = []
    for i, token in enumerate(tokens):
        # print(token, token.isidentifier)
        if token.isidentifier() and not is_java_keyword(token):
            # Check if it's likely a function (followed by a '(')
            if i + 1 < len(tokens) and tokens[i + 1] == '(':
                if token not in functions:
                    functions[token] = f'fn{fn_count}'
                    fn_count += 1
                normalized_tokens.append(functions[token])
            else:
                if token not in variables:
                    variables[token] = f'var{var_count}'
                    var_count += 1
                normalized_tokens.append(variables[token])
        else:
            normalized_tokens.append(token)

    return normalized_tokens

def main_java(inp, rm_comments, normalise):
    """
    output: str
        the input Java code as a string
    rm_comments: bool
        flag to indicate whether comments should be removed
    normalise: bool
        flag to indicate whether to normalise identifiers in the code
    """

    if rm_comments:
        inp = remove_comments_java(inp)

    tokens = tokenize_java_code(inp.lower())

    if normalise and rm_comments:
        tokens = normalize_identifiers_java(tokens)

    return tokens
