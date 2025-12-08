/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_variable_operation

import org.checkerframework.checker.index.qual.NonNegative;

public class RefineSubtrahend {

    void cases(int[] a, @NonNegative int l) {
        switch(a.length - l) {
            case 1:
                int x = a[l];
                break;
            case 2:
                int y = a[l + 1];
                break;
        }
    }
}
