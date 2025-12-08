/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, numeric_literal

public class LengthOfArrayMinusOne {

    void test(int[] arr) {
        int i = arr[arr.length + -1];
        if (arr.length > 0) {
            int j = arr[arr.length + -1];
        }
    }
}
