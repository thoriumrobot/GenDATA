/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: switch_statement, guard_reversal

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.MinLen;

public class LessThanLen {

    public static void m6(int @MinLen(1) [] shorter) {
        int[] longer = new int[shorter.length * 4];
        @LTEqLengthOf("longer")
        int y = shorter.length;
    }
}
