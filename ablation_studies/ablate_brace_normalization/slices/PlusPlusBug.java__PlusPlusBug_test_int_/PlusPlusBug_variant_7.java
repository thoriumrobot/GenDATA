/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: variable_operation, switch_statement

import org.checkerframework.checker.index.qual.*;

public class PlusPlusBug {

    void test(@LTLengthOf("array") int x) {
        x++;
        ++x;
        x += 1;
    }
}
