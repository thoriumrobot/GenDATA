/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_guard_reversal

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.MinLen;

public class UncheckedMinLen {

    void addToPositiveOK(@NonNegative int l, Object v) {
        Object[] o = new Object[l + 1];
        o[99] = v;
    }
}
