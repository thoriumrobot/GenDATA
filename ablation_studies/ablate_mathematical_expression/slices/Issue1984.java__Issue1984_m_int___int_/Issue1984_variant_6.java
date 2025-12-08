/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_variable_operation

import org.checkerframework.common.value.qual.IntRange;

public class Issue1984 {

    public int m(int[] a, @IntRange(from = 0, to = 12) int i) {
        return a[i];
    }
}
