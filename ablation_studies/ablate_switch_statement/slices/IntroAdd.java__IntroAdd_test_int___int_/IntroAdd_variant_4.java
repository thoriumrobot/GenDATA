/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: variable_operation, guard_reversal

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class IntroAdd {

    void test(int[] arr, @LTLengthOf({ "#1" }) int a) {
        @LTLengthOf({ "arr" })
        int c = 1 + a;
        @LTEqLengthOf({ "arr" })
        int c1 = 1 + a;
        @LTLengthOf({ "arr" })
        int d = a + 0;
        @LTLengthOf({ "arr" })
        int e = a + (-7);
        @LTLengthOf({ "arr" })
        int f = 7 + a;
    }
}
