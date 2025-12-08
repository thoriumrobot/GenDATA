/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_loop_conversion

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class ArrayConstructionPositiveLength {

    public void makeArray(@Positive int max_values) {
        String @MinLen(1) [] a = new String[max_values];
    }
}
