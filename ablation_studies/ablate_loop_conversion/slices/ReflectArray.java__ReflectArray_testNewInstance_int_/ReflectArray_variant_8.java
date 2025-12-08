/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_logical_expression, attempted_guard_reversal

import java.lang.reflect.Array;
import org.checkerframework.common.value.qual.MinLen;

public class ReflectArray {

    void testNewInstance(int i) {
        Array.newInstance(Object.class, i);
        if (i >= 0) {
            Array.newInstance(Object.class, i);
        }
    }
}
