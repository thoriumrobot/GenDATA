/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_variable_operation

import org.checkerframework.common.value.qual.MinLen;

public class LessThanOrEqualTransfer {

    void lte_bad_check(int[] a) {
        if (1 <= a.length) {
            int @MinLen(2) [] b = a;
        }
    }
}
