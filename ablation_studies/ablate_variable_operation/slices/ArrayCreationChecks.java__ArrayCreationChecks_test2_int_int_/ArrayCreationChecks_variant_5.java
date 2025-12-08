/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, string_concatenation

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test2(@NonNegative int x, @Positive int y) {
        int[] newArray = new int[String.valueOf(x + y)];
        @IndexFor("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
}
