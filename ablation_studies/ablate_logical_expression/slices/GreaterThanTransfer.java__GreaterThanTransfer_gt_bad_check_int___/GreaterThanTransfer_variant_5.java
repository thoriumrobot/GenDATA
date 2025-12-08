/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_string_concatenation

import org.checkerframework.common.value.qual.MinLen;

public class GreaterThanTransfer {

    void gt_bad_check(int[] a) {
        if (a.length > 0) {
            int @MinLen(2) [] b = a;
        }
    }
}
