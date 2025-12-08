/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_guard_reversal, attempted_ternary_operator

public class PreAndPostDec {

    void pre2(int[] args) {
        int ii = 0;
        while ((ii < args.length)) {
            ii++;
            int m = args[ii];
        }
    }
}
