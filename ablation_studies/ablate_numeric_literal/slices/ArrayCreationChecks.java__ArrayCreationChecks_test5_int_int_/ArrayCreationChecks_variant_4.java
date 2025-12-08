/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_guard_reversal

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
