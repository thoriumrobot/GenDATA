/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_mathematical_expression

public class StringMethods {

    void testCodePointBefore(String s) {
        s.codePointBefore(0);
        if (s.length() > 0) {
            s.codePointBefore(s.length());
        }
    }
}
