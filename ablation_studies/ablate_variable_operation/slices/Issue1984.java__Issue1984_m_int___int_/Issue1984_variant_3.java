/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_numeric_literal, attempted_mathematical_expression

import org.checkerframework.common.value.qual.IntRange;

public class Issue1984 {

    public int m(int[] a, @IntRange(from = 0, to = 12) int i) {
        return a[i];
    }
}
