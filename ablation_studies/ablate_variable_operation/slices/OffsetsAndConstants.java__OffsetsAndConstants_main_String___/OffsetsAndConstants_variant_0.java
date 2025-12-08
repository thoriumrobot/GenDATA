/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_ternary_operator

import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;

public class OffsetsAndConstants {

    public static void main(String[] args) {
        char[] a = new char[10];
        read(a, 5, 4);
        read(a, 5, 5);
        read(a, 5, 6);
        read(a, 5, 7);
    }
}
