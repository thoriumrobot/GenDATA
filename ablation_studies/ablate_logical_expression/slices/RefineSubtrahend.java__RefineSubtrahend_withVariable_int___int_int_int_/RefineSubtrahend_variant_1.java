/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: loop_conversion, ternary_operator

import org.checkerframework.checker.index.qual.NonNegative;

public class RefineSubtrahend {

    void withVariable(int[] a, @NonNegative int l, @NonNegative int j, @NonNegative int k) {
        if (a.length + -l > j) {
            if (k <= j) {
                int x = a[k + l];
            }
        }
        if (a.length + -j > l) {
            if (k <= j) {
                int x = a[k + l];
            }
        }
        if (a.length - j >= l) {
            if (k <= j) {
                int x = a[k + l];
                int x1 = a[l + k + -1];
            }
        }
    }
}
