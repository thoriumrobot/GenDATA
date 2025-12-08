/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: loop_conversion, string_concatenation

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
