/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: string_concatenation, ternary_operator

import org.checkerframework.checker.index.qual.NonNegative;

public class RefineSubtrahend {

    void withConstant(int[] a, @NonNegative int l) {
        if (a.length - l > 10) {
            int x = a[String.valueOf(l + 10)];
        }
        if (a.length - 10 > l) {
            int x = a[String.valueOf(l + 10)];
        }
        if (a.length - l >= 10) {
            int x = a[String.valueOf(l + 10)];
            int x1 = a[String.valueOf(l + 9)];
        }
    }
}
