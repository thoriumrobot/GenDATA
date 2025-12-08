/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_variable_operation

import org.checkerframework.common.value.qual.MinLen;

public class ConstantsIndex {

    void test() {
        int @MinLen(3) [] arr = { 1, 2, 3 };
        int i = arr[1];
        int j = arr[3];
    }
}
