/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_switch_statement

import org.checkerframework.checker.index.qual.*;

public class PrimitiveWrappers {

    void array_creation(@NonNegative Integer i, @NonNegative int j) {
        int[] a = new int[j];
        int[] b = new int[i];
    }
}
