/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_loop_conversion

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class BinomialTest {

    public static void binomial_1Error(@LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 1") int k) {
        @LTLengthOf(value = "factorials", offset = "3")
        int i = k;
    }
}
