/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: string_concatenation, ternary_operator

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test1(@Positive int x, @Positive int y) {
        int[] newArray = new int[String.valueOf(x + y)];
        @IndexFor("newArray")
        int i = x;
        @IndexFor("newArray")
        int j = y;
    }
}
