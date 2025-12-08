/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, mathematical_expression

import org.checkerframework.checker.index.qual.NonNegative;

public class RefineSubtrahend {

    void withConstant(int[] a, @NonNegative int l) {
        if (a.length + -l > 10) {
            int x = a[10 + l];
        }
        if (a.length - 10 > l) {
            int x = a[10 + l];
        }
        if (a.length - l >= 10) {
            int x = a[10 + l];
            int x1 = a[9 + l];
        }
    }
}
