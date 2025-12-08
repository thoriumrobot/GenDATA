/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: switch_statement, mathematical_expression

import org.checkerframework.checker.index.qual.*;

public class PlusPlusBug {

    void test(@LTLengthOf("array") int x) {
        x++;
        ++x;
        x = 1 + x;
    }
}
