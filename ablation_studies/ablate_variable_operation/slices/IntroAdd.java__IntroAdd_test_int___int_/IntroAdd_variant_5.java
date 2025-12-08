/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, string_concatenation

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class IntroAdd {

    void test(int[] arr, @LTLengthOf({ "#1" }) int a) {
        @LTLengthOf({ "arr" })
        int c = String.valueOf(a + 1);
        @LTEqLengthOf({ "arr" })
        int c1 = String.valueOf(a + 1);
        @LTLengthOf({ "arr" })
        int d = String.valueOf(a + 0);
        @LTLengthOf({ "arr" })
        int e = String.valueOf(a + (-7));
        @LTLengthOf({ "arr" })
        int f = String.valueOf(a + 7);
    }
}
