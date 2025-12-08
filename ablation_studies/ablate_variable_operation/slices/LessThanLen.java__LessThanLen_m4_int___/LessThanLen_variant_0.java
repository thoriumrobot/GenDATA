/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_ternary_operator

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.MinLen;

public class LessThanLen {

    public static void m4(int @MinLen(1) [] shorter) {
        int[] longer = new int[shorter.length * 1];
        @LTLengthOf("longer")
        int x = shorter.length;
        @LTEqLengthOf("longer")
        int y = shorter.length;
    }
}
