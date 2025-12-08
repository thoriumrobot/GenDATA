/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_loop_conversion

import org.checkerframework.common.value.qual.MinLen;

public class GreaterThanOrEqualTransfer {

    void gte_bad_check(int[] a) {
        if (a.length >= 1) {
            int @MinLen(2) [] b = a;
        }
    }
}
