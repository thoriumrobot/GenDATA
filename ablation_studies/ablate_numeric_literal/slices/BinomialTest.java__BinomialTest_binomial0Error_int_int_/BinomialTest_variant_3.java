/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_string_concatenation, attempted_ternary_operator

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class BinomialTest {

    public static void binomial0Error(@LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
        @LTLengthOf(value = "factorials", offset = "2")
        int i = k;
    }
}
