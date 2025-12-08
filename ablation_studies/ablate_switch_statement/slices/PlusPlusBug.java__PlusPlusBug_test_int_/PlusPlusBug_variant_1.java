/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_ternary_operator

import org.checkerframework.checker.index.qual.*;

public class PlusPlusBug {

    void test(@LTLengthOf("array") int x) {
        x++;
        ++x;
        x = x + 1;
    }
}
