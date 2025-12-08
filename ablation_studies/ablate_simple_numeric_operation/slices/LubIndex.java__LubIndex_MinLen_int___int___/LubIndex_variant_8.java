/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_switch_statement, attempted_logical_expression

import org.checkerframework.common.value.qual.BottomVal;
import org.checkerframework.common.value.qual.MinLen;

public class LubIndex {

    public static void MinLen(int @MinLen(10) [] arg, int @MinLen(4) [] arg2) {
        int[] arr;
        if (true) {
            arr = arg;
        } else {
            arr = arg2;
        }
        int @MinLen(10) [] res = arr;
        int @MinLen(4) [] res2 = arr;
        int @BottomVal [] res3 = arr;
    }
}
