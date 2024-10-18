import ply.lex as lex

# List of token names
tokens = [
    'IDENTIFIER', 'NUMBER', 'STRING_LITERAL', 'CHAR_LITERAL',
    # Operators
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
    'EQUALS', 'PLUSEQUAL', 'MINUSEQUAL', 'TIMESEQUAL', 'DIVEQUAL',
    'PLUSPLUS', 'MINUSMINUS',
    'LNOT', 'LAND', 'LOR',
    'BAND', 'BOR', 'BXOR', 'BNOT', 'LSHIFT', 'RSHIFT',
    'LT', 'GT', 'LE', 'GE', 'EQ', 'NE',
    # Delimiters
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
    'COMMA', 'DOT', 'SEMICOLON', 'COLON', 'QUESTION',
    # Preprocessor
    'PPHASH',
    # Comments (ignored)
]

# Operators
t_PLUS          = r'\+'
t_MINUS         = r'-'
t_TIMES         = r'\*'
t_DIVIDE        = r'/'
t_MOD           = r'%'
t_EQUALS        = r'='
t_PLUSEQUAL     = r'\+='
t_MINUSEQUAL    = r'-='
t_TIMESEQUAL    = r'\*='
t_DIVEQUAL      = r'/='
t_PLUSPLUS      = r'\+\+'
t_MINUSMINUS    = r'--'
t_LNOT          = r'!'
t_LAND          = r'&&'
t_LOR           = r'\|\|'
t_BAND          = r'&'
t_BOR           = r'\|'
t_BXOR          = r'\^'
t_BNOT          = r'~'
t_LSHIFT        = r'<<'
t_RSHIFT        = r'>>'
t_LT            = r'<'
t_GT            = r'>'
t_LE            = r'<='
t_GE            = r'>='
t_EQ            = r'=='
t_NE            = r'!='
t_DOT          = r'\.'
t_COLON        = r':'
t_QUESTION     = r'\?'
t_LBRACKET     = r'\['
t_RBRACKET     = r'\]'

# Delimiters
t_LPAREN        = r'\('
t_RPAREN        = r'\)'
t_LBRACE        = r'\{'
t_RBRACE        = r'\}'
t_COMMA         = r','

t_SEMICOLON     = r';'
# Whitespace and tabs (ignored)
t_ignore = ' \t'



# Preprocessor
def t_PPHASH(t):
    r'\#.*'
    pass  # Ignore or handle as needed

# Comments (ignored)
def t_COMMENT(t):
    r'(/\*(.|\n)*?\*/)|(//.*)'
    pass

# String literal
def t_STRING_LITERAL(t):
    r'"([^"\\]|\\.)*"'
    return t

# Character literal
def t_CHAR_LITERAL(t):
    r'\'([^\'\\]|\\.)\''
    return t

# Identifier
def t_IDENTIFIER(t):
    r'[A-Za-z_][A-Za-z0-9_]*'
    return t

# Number
def t_NUMBER(t):
    r'\d+(\.\d+)?([eE][+-]?\d+)?'
    return t

# Newlines
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Error handling
def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)

lexer = lex.lex()

def tokenize_code(code):

    lexer.input(code)
    tokens = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens.append(tok)
    return tokens

def is_keyword(token):
    cpp_keywords = set([
        'if', 'else', 'while', 'for', 'return', 'int', 'float', 'double', 'char', 'bool', 'void',
        'class', 'struct', 'namespace', 'using', 'include', 'true', 'false', 'switch', 'case', 'break',
        'continue', 'default', 'public', 'private', 'protected', 'virtual', 'static', 'new', 'delete',
        'try', 'catch', 'throw', 'this', 'const', 'sizeof', 'typedef', 'template'
    ])
    return token in cpp_keywords

def normalise_identifiers(tokens):
    var_count = 1
    fn_count = 1
    variables = {}
    functions = {}
    normalized_tokens = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == 'IDENTIFIER' and not is_keyword(token.value):
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            if next_token and next_token.type == 'LPAREN':
                if token.value not in functions:
                    functions[token.value] = f'fn{fn_count}'
                    fn_count += 1
                token.value = functions[token.value]
            else:
                if token.value not in variables:
                    variables[token.value] = f'var{var_count}'
                    var_count += 1
                token.value = variables[token.value]
        normalized_tokens.append(token)
        i += 1
    return normalized_tokens

def get_string(tokens):
    ls = [token.value for token in tokens]
    return ' '.join(ls)


