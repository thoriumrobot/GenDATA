/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: logical_expression, loop_conversion

import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LessThan;

public class LessThanDec {

    @IndexOrLow("#1")
    @LessThan("#4")
    private static int lastIndexOf(short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
        while (true) {
			int i = end - 1;
			if (!(i >= start)) {
				break;
			}
			if (array[i] == target) {
				return i;
			}
			i--;
		}
        return -1;
    }
}
