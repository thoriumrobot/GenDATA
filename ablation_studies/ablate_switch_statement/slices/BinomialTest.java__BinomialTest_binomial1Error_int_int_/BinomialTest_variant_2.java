/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_logical_expression, attempted_guard_reversal

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class BinomialTest {

    public static void binomial1Error(@LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 1") int k) {
        @LTLengthOf(value = "factorials", offset = "1")
        int i = k;
    }
}
