/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_numeric_literal, attempted_mathematical_expression

import java.lang.reflect.Array;
import org.checkerframework.common.value.qual.MinLen;

public class ReflectArray {

    void testMinLen(Object @MinLen(1) [] a) {
        Array.get(a, 0);
        Array.get(a, 1);
    }
}
