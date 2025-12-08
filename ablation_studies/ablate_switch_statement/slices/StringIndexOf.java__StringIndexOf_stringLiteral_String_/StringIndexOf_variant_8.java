/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_logical_expression, attempted_loop_conversion

public class StringIndexOf {

    public static String stringLiteral(String l) {
        int i = l.indexOf("constant");
        if (i != -1) {
            return l.substring(0, i) + l.substring(i + "constant".length());
        }
        return l.substring(0, i) + l.substring(i + "constant".length());
    }
}
