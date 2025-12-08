/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_loop_conversion

import org.checkerframework.checker.index.qual.HasSubsequence;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LessThan;

public class InvalidSubsequence {

    void assignE(int[] d) {
        e = d;
    }
}
