/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_string_concatenation, attempted_ternary_operator

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class BinomialTest {

    public static long binomial(@NonNegative @LTLengthOf("BinomialTest.factorials") int n, @NonNegative @LessThan("#1 + 1") int k) {
        return factorials[k];
    }
}
