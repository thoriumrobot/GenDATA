/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_variable_operation

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.SameLen;
import org.checkerframework.common.value.qual.MinLen;

public class ArrayWrapper {

    public void set(@IndexFor("this") int index, T obj) {
        delegate[index] = obj;
    }
}
