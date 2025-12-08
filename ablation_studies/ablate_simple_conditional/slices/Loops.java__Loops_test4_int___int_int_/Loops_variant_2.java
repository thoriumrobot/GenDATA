/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_switch_statement, attempted_loop_conversion

import org.checkerframework.checker.index.qual.LTLengthOf;

public class Loops {

    public void test4(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        while (flag) {
            offset++;
            offset += 1;
            offset2 += offset;
        }
    }
}
