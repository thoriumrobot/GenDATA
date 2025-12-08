/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_guard_reversal

import org.checkerframework.checker.index.qual.LTLengthOf;

public class Loops {

    public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        while (flag) {
            offset++;
        }
    }
}
