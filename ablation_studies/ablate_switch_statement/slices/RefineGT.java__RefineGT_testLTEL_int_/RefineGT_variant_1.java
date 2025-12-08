/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: loop_conversion, ternary_operator

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineGT {

    void testLTEL(@LTEqLengthOf("arr") int test) {
        @LTEqLengthOf("arr")
        int a = Integer.parseInt("1");
        @LTEqLengthOf("arr")
        int a3 = Integer.parseInt("3");
        int b = 2;
        if (test > b) {
            @LTLengthOf("arr")
            int c = b;
        }
        @LTLengthOf("arr")
        int c1 = b;
        if (!(a > b)){@LTLengthOf("arr") int d=b;} else {int potato=7;}
    }
}
