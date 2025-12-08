/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_logical_expression

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
