/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_variable_operation, attempted_switch_statement

import org.checkerframework.checker.index.qual.*;

public class BasicSubsequence {

    void test3(@NonNegative @LessThan("y") int x1, int[] a) {
        x = x1;
        b = a;
    }
}
