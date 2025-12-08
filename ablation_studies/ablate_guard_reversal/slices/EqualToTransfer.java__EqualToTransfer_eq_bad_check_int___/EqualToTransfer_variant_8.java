/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_logical_expression, attempted_loop_conversion

import org.checkerframework.common.value.qual.MinLen;

public class EqualToTransfer {

    void eq_bad_check(int[] a) {
        if (1 == a.length) {
            int @MinLen(2) [] b = a;
        }
    }
}
