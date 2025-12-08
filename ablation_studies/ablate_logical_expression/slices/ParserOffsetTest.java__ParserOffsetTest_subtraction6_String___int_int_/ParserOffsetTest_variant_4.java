/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: variable_operation, guard_reversal

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class ParserOffsetTest {

    public void subtraction6(String[] a, int i, int j) {
        if (i - j < a.length - 1) {
            @IndexFor("a")
            int k = i + -j;
            @IndexFor("a")
            int k1 = i;
        }
    }
}
