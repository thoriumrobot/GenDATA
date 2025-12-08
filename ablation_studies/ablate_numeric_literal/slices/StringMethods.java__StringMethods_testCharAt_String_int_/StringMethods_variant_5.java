/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_logical_expression

public class StringMethods {

    void testCharAt(String s, int i) {
        s.charAt(i);
        s.codePointAt(i);
        if (i >= 0 && i < s.length()) {
            s.charAt(i);
            s.codePointAt(i);
        }
    }
}
