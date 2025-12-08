/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_loop_conversion

import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.PolyUpperBound;

public class UBPoly {

    public static void access(char[] a, @NonNegative @LTLengthOf("#1") int j) {
        char c = a[j];
    }
}
