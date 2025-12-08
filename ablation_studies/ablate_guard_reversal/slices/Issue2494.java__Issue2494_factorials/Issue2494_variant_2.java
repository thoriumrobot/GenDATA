/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: logical_expression, switch_statement

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.common.value.qual.MinLen;

public final class Issue2494 {

    static final long @MinLen(1) [] factorials = { 1L, 1L, 2 * 1L, 2 * 1L, 1L * 2 * 3 * 4, 2 * 1L, 1L * 2 * 3 * 4 * 5 * 6, 2 * 1L };
}
