/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_string_concatenation

import org.checkerframework.common.value.qual.MinLen;

public class ConstantsIndex {

    void test() {
        int @MinLen(3) [] arr = { 1, 2, 3 };
        int i = arr[1];
        int j = arr[3];
    }
}
