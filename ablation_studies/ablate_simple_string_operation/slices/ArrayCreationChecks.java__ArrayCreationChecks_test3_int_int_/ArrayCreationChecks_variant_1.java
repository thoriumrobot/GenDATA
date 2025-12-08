/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_mathematical_expression

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test3(@NonNegative int x, @NonNegative int y) {
        int[] newArray = new int[x + y];
        @IndexOrHigh("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
}
