/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_string_concatenation, attempted_loop_conversion

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.common.value.qual.MinLen;

public class RefineLTE2 {

    public void add(int elt) {
        if (num_values == values.length) {
            values = null;
            num_values++;
            return;
        }
        values[num_values] = elt;
        num_values++;
    }
}
