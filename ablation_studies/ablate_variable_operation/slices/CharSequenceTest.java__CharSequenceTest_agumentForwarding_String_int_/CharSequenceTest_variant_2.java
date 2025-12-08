/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_logical_expression, attempted_switch_statement

import java.io.IOException;
import java.io.StringWriter;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;
import org.checkerframework.common.value.qual.StringVal;

public class CharSequenceTest {

    void agumentForwarding(String s, @IndexOrHigh("#1") int i) {
        sink(s, i);
    }
}
