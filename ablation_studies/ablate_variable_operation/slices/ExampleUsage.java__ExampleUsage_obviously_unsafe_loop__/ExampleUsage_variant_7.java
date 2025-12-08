/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: ternary_operator, loop_conversion

public class ExampleUsage {

    void obviously_unsafe_loop() {
        int[] arr = new int[5];
        int k;
        while (true) {
			int i = -1;
			k = arr[i];
			if (!(i < 5)) {
				break;
			}
			i++;
		}
    }
}
