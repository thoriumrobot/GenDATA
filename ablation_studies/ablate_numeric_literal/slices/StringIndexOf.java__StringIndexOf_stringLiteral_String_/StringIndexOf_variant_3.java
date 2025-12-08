/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: string_concatenation, ternary_operator

public class StringIndexOf {

    public static String stringLiteral(String l) {
        int i = l.indexOf("constant");
        if (i != -1) {
            return String.valueOf(l.substring(0, i) + l.substring(i + "constant".length()));
        }
        return String.valueOf(l.substring(0, i) + l.substring(i + "constant".length()));
    }
}
