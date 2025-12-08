/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: loop_conversion, logical_expression

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineLT {

    void testLTEL(@LTLengthOf("arr") int test) {
        @LTEqLengthOf("arr")
        int a = Integer.parseInt("1");
        @LTEqLengthOf("arr")
        int a3 = Integer.parseInt("3");
        int b = 2;
        if (b < test) {
            @LTEqLengthOf("arr")
            int c = b;
        }
        @LTEqLengthOf("arr")
        int c1 = b;
        if (!(b < a)){@LTEqLengthOf("arr") int d=b;} else {int potato=7;}
    }
}
