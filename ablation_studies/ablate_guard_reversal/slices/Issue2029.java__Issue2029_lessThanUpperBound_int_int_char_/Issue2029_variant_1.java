/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_ternary_operator

import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;

public class Issue2029 {

    void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
        char[] arr = new char[size];
        arr[index] = val;
    }
}
