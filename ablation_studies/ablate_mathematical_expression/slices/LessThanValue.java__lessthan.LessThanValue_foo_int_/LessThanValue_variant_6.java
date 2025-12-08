/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_variable_operation

package lessthan;

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class LessThanValue {

    @LessThan("#1")
    @NonNegative
    int foo(int in) {
        throw new RuntimeException();
    }
}
