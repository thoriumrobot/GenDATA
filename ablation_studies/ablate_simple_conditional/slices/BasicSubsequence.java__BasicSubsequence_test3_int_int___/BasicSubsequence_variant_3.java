/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_string_concatenation, attempted_loop_conversion

import org.checkerframework.checker.index.qual.*;

public class BasicSubsequence {

    void test3(@NonNegative @LessThan("y") int x1, int[] a) {
        x = x1;
        b = a;
    }
}
