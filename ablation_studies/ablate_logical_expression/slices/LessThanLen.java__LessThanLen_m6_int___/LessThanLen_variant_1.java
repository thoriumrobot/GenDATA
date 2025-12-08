/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_ternary_operator

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.MinLen;

public class LessThanLen {

    public static void m6(int @MinLen(1) [] shorter) {
        int[] longer = new int[4 * shorter.length];
        @LTEqLengthOf("longer")
        int y = shorter.length;
    }
}
