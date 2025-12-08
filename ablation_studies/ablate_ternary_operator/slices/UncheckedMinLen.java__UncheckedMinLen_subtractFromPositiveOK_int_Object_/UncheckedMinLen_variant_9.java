/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: switch_statement, loop_conversion

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.MinLen;

public class UncheckedMinLen {

    void subtractFromPositiveOK(@Positive int l, Object v) {
        Object @MinLen(100) [] o = new Object[l + -1];
        o[99] = v;
    }
}
