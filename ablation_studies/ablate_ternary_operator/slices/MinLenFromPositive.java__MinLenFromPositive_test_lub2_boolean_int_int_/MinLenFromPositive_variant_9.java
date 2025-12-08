/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: switch_statement, loop_conversion

import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.*;

public class MinLenFromPositive {

    void test_lub2(boolean flag, @Positive int x, @IntRange(from = -1, to = 11) int y) {
        int z;
        if (!(flag)) {
			z = y;
		} else {
			z = x;
		}
        @Positive
        int q = z;
        @IntRange(from = -1)
        int w = z;
    }
}
