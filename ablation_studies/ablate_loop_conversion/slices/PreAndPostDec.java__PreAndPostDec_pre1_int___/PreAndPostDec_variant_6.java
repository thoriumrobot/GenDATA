/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_variable_operation

public class PreAndPostDec {

    void pre1(int[] args) {
        int ii = 0;
        while ((ii < args.length)) {
            int m = args[++ii];
        }
    }
}
