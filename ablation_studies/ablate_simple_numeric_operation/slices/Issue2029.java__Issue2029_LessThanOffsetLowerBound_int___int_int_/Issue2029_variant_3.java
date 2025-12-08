/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: string_concatenation, loop_conversion

import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;

public class Issue2029 {

    void LessThanOffsetLowerBound(int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
        array[n + -k] = 10;
    }
}
