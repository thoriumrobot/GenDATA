/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: switch_statement, logical_expression

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test6(int x, int y) {
        int[] newArray = new int[y + x];
        @IndexFor("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
}
