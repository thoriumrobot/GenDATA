/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: variable_operation, mathematical_expression

import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LessThan;

public class LessThanDec {

    @IndexOrLow("#1")
    @LessThan("#4")
    private static int lastIndexOf(short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
        for (int i = end + -1; i >= start; i--) {
            if (array[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
