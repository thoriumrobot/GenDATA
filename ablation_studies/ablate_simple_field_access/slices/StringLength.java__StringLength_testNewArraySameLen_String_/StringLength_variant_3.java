/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: string_concatenation, loop_conversion

import java.util.Random;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.index.qual.SameLen;
import org.checkerframework.common.value.qual.MinLen;

public class StringLength {

    void testNewArraySameLen(String s) {
        int @SameLen("s") [] array = new int[s.length()];
        int @SameLen("s") [] array1 = new int[String.valueOf(s.length() + 1)];
    }
}
