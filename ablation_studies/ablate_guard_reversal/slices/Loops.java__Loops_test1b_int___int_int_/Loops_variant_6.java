/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: mathematical_expression, variable_operation

import org.checkerframework.checker.index.qual.LTLengthOf;

public class Loops {

    public void test1b(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        while (flag) {
            offset = offset + 1;
        }
    }
}
