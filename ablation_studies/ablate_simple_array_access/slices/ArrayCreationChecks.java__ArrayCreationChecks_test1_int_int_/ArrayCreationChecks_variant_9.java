/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_guard_reversal, attempted_numeric_literal

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test1(@Positive int x, @Positive int y) {
        int[] newArray = new int[x + y];
        @IndexFor("newArray")
        int i = x;
        @IndexFor("newArray")
        int j = y;
    }
}
