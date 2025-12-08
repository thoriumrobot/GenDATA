/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_switch_statement, attempted_guard_reversal

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LTOMLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.common.value.qual.IntVal;

public class RefineNeqLength {

    void refineNeqLengthMOne(int[] array, @IndexFor("#1") int i) {
        if (i != array.length - 1) {
            refineNeqLengthMTwo(array, i);
            refineNeqLengthMThree(array, i);
        }
    }
}
