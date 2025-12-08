/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_variable_operation

import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.*;

public class MinLenFromPositive {

    @Positive
    int id(@Positive int x) {
        return x;
    }
}
