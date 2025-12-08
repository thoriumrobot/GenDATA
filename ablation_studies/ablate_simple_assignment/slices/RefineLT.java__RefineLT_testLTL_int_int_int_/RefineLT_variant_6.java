/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_variable_operation

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineLT {

    void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        int b = 2;
        if (b < test) {
            @LTLengthOf("arr")
            int c = b;
        }
        @LTLengthOf("arr")
        int c1 = b;
        if (b < a3) {
            int potato = 7;
        } else {
            @LTLengthOf("arr")
            int d = b;
        }
    }
}
