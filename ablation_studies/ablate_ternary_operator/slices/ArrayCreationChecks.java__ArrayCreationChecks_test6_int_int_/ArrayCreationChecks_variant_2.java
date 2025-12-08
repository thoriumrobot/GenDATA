/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_logical_expression, attempted_switch_statement

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test6(int x, int y) {
        int[] newArray = new int[x + y];
        @IndexFor("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
}
