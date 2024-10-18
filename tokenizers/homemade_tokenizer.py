import re


def remove_comments(code):
    """Removes both single-line and multi-line comments from the code."""
    # Remove comments using regex (dotall mode allows matching newlines inside comments)
    comment_pattern = r'//.*?$|/\*.*?\*/'
    no_comments_code = re.sub(comment_pattern, '', code, flags=re.DOTALL | re.MULTILINE)
    return no_comments_code


def tokenize_code(code):
    # List of multi-character operators that should not be split
    multi_char_operators = ['==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '&&', '||', '++', '--']

    # Regex pattern to match variables, numbers, multi-character operators, and symbols
    pattern = r'(\w+|==|!=|<=|>=|\+=|-=|\*=|/=|&&|\|\||\+\+|--|[+\-*/=(){};><,\]\[])'
    tokens = []
    for line in code.splitlines():
        line_tokens = re.findall(pattern, line.strip())
        tokens.extend(line_tokens)
    return tokens


def is_keyword(token):
    cpp_keywords = [
        'if', 'else', 'while', 'for', 'return', 'int', 'float', 'double', 'char', 'bool', 'void',
        'class', 'struct', 'namespace', 'using', 'include', 'true', 'false', 'switch', 'case', 'break',
        'continue', 'default', 'public', 'private', 'protected', 'virtual', 'static', 'new', 'delete',
        'try', 'catch', 'throw', 'this', 'const', 'sizeof', 'typedef', 'template'
    ]
    return token in cpp_keywords


def normalize_identifiers(tokens):
    var_count = 1
    fn_count = 1
    variables = {}
    functions = {}

    normalized_tokens = []
    for i, token in enumerate(tokens):
        if token.isidentifier() and not is_keyword(token):
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

def tokenize_list(ls):
    temp_str = ""
    for item in ls:
        tokens = tokenize_code(item)
        temp_str += tokens
    print(temp_str)
    tokens = tokenize_code(temp_str.lower())
    return tokens

def normalise_string(inp):
    tokens = tokenize_code(inp.lower())

    tokens = normalize_identifiers(tokens)
    output = ""
    for token in tokens:
        output += token + " "
    return output

def main(output, rm_comments, normalise):
    """
    output: str
        the input... an unparsed c++ code file
    rm_comments: bool
        flag to indicate whether comments should be removed
    normalise: bool
        flag to indicate whether to normalise identifiers in the code
    """

    if rm_comments:
        output = remove_comments(output)

    tokens = tokenize_code(output.lower())

    if normalise and rm_comments:
        tokens = normalize_identifiers(tokens)

    return tokens
