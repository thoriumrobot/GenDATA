/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_switch_statement

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
