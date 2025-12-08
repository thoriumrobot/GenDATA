/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: logical_expression, guard_reversal

import org.checkerframework.checker.index.qual.*;

public class IntroAnd {

    void test_ubc_and(@IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        int x = a[k & i];
        int x1 = a[k & i];
        int y = a[k & j];
        if (j > -1) {
            int z = a[k & j];
        }
        int w = a[k & m];
        if (m < a.length) {
            int u = a[k & m];
        }
    }
}
