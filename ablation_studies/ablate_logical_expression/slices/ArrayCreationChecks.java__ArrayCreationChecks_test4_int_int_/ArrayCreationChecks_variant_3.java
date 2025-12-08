/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_numeric_literal, attempted_ternary_operator

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test4(@GTENegativeOne int x, @NonNegative int y) {
        int[] newArray = new int[x + y];
        @LTEqLengthOf("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
}
