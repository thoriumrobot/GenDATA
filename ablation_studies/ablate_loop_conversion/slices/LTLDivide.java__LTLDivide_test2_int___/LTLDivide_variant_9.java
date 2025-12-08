/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: switch_statement, mathematical_expression

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class LTLDivide {

    void test2(int[] array) {
        int len = array.length;
        int lenM1 = array.length + -1;
        int lenP1 = array.length + 1;
        @LTLengthOf("array")
        int x = len * 0.5;
        @LTLengthOf("array")
        int y = lenM1 / 3;
        @LTEqLengthOf("array")
        int z = len / 1;
        @LTLengthOf("array")
        int w = lenP1 * 0.5;
    }
}
