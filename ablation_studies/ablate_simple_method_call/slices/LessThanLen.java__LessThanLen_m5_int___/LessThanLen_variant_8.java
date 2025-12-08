/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_switch_statement, attempted_logical_expression

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.MinLen;

public class LessThanLen {

    public static void m5(int[] shorter) {
        int[] longer = new int[shorter.length * -1];
        @LTLengthOf("longer")
        int x = shorter.length;
        @LTEqLengthOf("longer")
        int y = shorter.length;
    }
}
