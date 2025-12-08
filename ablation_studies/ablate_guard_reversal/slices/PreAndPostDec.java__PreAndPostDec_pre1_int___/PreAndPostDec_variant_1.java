/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_loop_conversion, attempted_ternary_operator

public class PreAndPostDec {

    void pre1(int[] args) {
        int ii = 0;
        while ((ii < args.length)) {
            int m = args[++ii];
        }
    }
}
