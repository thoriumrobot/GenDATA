/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_switch_statement

public class StringMethods {

    void testCodePointBefore(String s) {
        s.codePointBefore(0);
        if (s.length() > 0) {
            s.codePointBefore(s.length());
        }
    }
}
