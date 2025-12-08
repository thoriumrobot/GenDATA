/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_variable_operation

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.common.value.qual.MinLen;

public class IndexForTest {

    void test2(@IndexFor("this.array") int i) {
        int x = array[i];
    }
}
