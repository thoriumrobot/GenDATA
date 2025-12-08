/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_switch_statement, attempted_guard_reversal

import org.checkerframework.common.value.qual.MinLen;

public class ConstantsIndex {

    void test() {
        int @MinLen(3) [] arr = { 1, 2, 3 };
        int i = arr[1];
        int j = arr[3];
    }
}
