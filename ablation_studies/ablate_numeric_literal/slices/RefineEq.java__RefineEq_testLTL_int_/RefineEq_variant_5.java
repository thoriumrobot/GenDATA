/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: loop_conversion, logical_expression

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineEq {

    void testLTL(@LTLengthOf("arr") int test) {
        @LTLengthOf("arr")
        int a = Integer.parseInt("1");
        int b = 1;
        if (!(test == b)){@LTLengthOf("arr") int e=b;} else {@LTLengthOf("arr") int c=b;}
        @LTLengthOf("arr")
        int d = b;
    }
}
