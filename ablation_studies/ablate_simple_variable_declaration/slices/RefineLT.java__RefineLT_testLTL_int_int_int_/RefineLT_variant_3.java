/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: string_concatenation, loop_conversion

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
        if (!(b < a3)){@LTLengthOf("arr") int d=b;} else {int potato=7;}
    }
}
