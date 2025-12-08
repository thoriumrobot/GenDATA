/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_variable_operation

public class StringIndexOf {

    public static String nocheck(String l, String s) {
        int i = l.indexOf(s);
        return l.substring(0, i) + l.substring(i + s.length());
    }
}
