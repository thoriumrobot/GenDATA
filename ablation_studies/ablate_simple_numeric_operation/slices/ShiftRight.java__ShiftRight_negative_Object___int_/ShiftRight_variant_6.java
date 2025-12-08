/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_variable_operation

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class ShiftRight {

    void negative(Object[] a, @LTLengthOf(value = "#1", offset = "100") int i) {
        @LTLengthOf(value = "#1", offset = "100")
        int q = i >> 2;
    }
}
