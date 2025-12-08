/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_switch_statement

import org.checkerframework.common.value.qual.MinLen;

public class GreaterThanOrEqualTransfer {

    void gte_bad_check(int[] a) {
        if (a.length >= 1) {
            int @MinLen(2) [] b = a;
        }
    }
}
