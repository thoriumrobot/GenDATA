/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: mathematical_expression, variable_operation

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class ParserOffsetTest {

    public void subtraction5(String[] a, int i) {
        if (1 + -i < a.length) {
            @IndexFor("a")
            int j = i;
        }
    }
}
