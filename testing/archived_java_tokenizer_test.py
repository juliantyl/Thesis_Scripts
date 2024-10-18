from tokenizers.homemade_tokenizer_java import main_java

java_code = """
public class Main {
    // This is a single-line comment
    /*
     * This is a multi-line comment
     */
    public static void main(String[] args) {
        int a = 5;
        int b = 10;
        System.out.println(a + b);
    }
}
"""

# Tokenize Java code and remove comments
tokens = main_java(java_code, rm_comments=True, normalise=True)
print(tokens)
