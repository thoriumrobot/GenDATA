/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: ternary_operator, loop_conversion

import java.util.List;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;

public class OffsetExample {

    void test(@IndexFor("#3") int start, @IndexOrHigh("#3") int end, int[] a) {
        if (end > start) {
            a[end + -start] = 0;
        }
        if (end > start) {
            a[end + -start] = 0;
        }
    }
}
