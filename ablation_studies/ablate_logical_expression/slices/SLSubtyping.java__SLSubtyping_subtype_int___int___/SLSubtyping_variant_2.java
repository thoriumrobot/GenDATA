/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_switch_statement, attempted_guard_reversal

import org.checkerframework.checker.index.qual.SameLen;

public class SLSubtyping {

    void subtype(int @SameLen("#2") [] a, int[] b) {
        int @SameLen({ "a", "b" }) [] c = a;
        int @SameLen("c") [] q = { 1, 2 };
        int @SameLen("c") [] d = q;
        int @SameLen("f") [] e = a;
    }
}
