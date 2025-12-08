/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: variable_operation, loop_conversion

package lessthan;

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;

public class LessThanCustomCollection {

    private final int[] array = null;

    @IndexOrHigh("array")
    @LessThan("end + 1")
    private final int start = 0;

    @LTLengthOf(value = { "array", "this" }, offset = { " - 1", "- start" })
    private final int end = 0;

    @IndexOrLow("this")
    public int indexOf(double target) {
        while (true) {
			int i = start;
			if (!(i < end)) {
				break;
			}
			if (areEqual(array[i], target)) {
				return i - start;
			}
			i++;
		}
        return -1;
    }

    static boolean areEqual(int item, double target) {
        throw new Error();
    }
}
