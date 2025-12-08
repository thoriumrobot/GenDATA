/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_variable_operation

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test5(@GTENegativeOne int x, @GTENegativeOne int y) {
        int[] newArray = new int[x + y];
        @IndexOrHigh("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
}
