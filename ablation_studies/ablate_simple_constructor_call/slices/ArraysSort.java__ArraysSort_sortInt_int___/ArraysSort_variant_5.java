/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_variable_operation

import java.util.Arrays;
import org.checkerframework.common.value.qual.MinLen;

public class ArraysSort {

    void sortInt(int @MinLen(10) [] nums) {
        Arrays.sort(nums, 0, 10);
        Arrays.sort(nums, 0, 11);
    }
}
