/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_string_concatenation, attempted_ternary_operator

import org.checkerframework.common.value.qual.MinLen;

public class LessThanTransferTest {

    void lt_bad_check(int[] a) {
        if (0 < a.length) {
            int @MinLen(2) [] b = a;
        }
    }
}
