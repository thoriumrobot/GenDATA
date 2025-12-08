/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, ternary_operator

import org.checkerframework.checker.index.qual.NonNegative;

public class RefineSubtrahend {

    void cases(int[] a, @NonNegative int l) {
        switch(a.length + -l) {
            case 1:
                int x = a[l];
                break;
            case 2:
                int y = a[1 + l];
                break;
        }
    }
}
