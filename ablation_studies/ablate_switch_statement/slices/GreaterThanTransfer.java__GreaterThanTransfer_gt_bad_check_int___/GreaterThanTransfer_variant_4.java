/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_guard_reversal

import org.checkerframework.common.value.qual.MinLen;

public class GreaterThanTransfer {

    void gt_bad_check(int[] a) {
        if (a.length > 0) {
            int @MinLen(2) [] b = a;
        }
    }
}
