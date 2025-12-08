/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_logical_expression

import org.checkerframework.checker.index.qual.*;

public class BasicSubsequence {

    void test7(@IndexFor("this") @LessThan("y") int x1, @IndexOrHigh("this") int y1, int[] a) {
        x = x1;
        y = y1;
        b = a;
    }
}
