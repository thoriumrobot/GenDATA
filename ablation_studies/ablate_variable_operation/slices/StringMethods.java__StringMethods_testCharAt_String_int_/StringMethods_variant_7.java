/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_loop_conversion

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
