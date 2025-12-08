/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_variable_operation

import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.*;

public class MinLenFromPositive {

    void test(@Positive int x) {
        int @MinLen(1) [] y = new int[x];
        @IntRange(from = 1)
        int z = x;
        @Positive
        int q = x;
    }
}
