/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: ternary_operator, switch_statement

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.MinLen;

public class LessThanLen {

    public static void m4(int @MinLen(1) [] shorter) {
        int[] longer = new int[1 * shorter.length];
        @LTLengthOf("longer")
        int x = shorter.length;
        @LTEqLengthOf("longer")
        int y = shorter.length;
    }
}
