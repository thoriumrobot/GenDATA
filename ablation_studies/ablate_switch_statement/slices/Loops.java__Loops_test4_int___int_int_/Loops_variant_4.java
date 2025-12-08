/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: variable_operation, guard_reversal

import org.checkerframework.checker.index.qual.LTLengthOf;

public class Loops {

    public void test4(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        while (flag) {
            offset++;
            offset = offset + 1;
            offset2 = offset2 + offset;
        }
    }
}
