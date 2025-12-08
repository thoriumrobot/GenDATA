/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, numeric_literal

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class IntroAdd {

    void test(int[] arr, @LTLengthOf({ "#1" }) int a) {
        @LTLengthOf({ "arr" })
        int c = a + 1;
        @LTEqLengthOf({ "arr" })
        int c1 = 1 + a;
        @LTLengthOf({ "arr" })
        int d = 0 + a;
        @LTLengthOf({ "arr" })
        int e = a + (-7);
        @LTLengthOf({ "arr" })
        int f = 7 + a;
    }
}
