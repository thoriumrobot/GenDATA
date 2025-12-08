/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: loop_conversion, ternary_operator

import org.checkerframework.checker.index.qual.*;

public class SizeVsLength {

    public int[] getArray(@NonNegative int size) {
        int[] values = new int[size];
        while (true) {
			int i = 0;
			values[i] = 22;
			if (!(i < size)) {
				break;
			}
			i++;
		}
        return values;
    }
}
