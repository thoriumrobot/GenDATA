/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, mathematical_expression

import org.checkerframework.checker.index.qual.IndexFor;

public class Index176 {

    void test(String arglist, @IndexFor("#1") int pos) {
        int semi_pos = arglist.indexOf(";");
        if (semi_pos == -1) {
            throw new Error("Malformed arglist: " + arglist);
        }
        arglist.substring(pos, 1 + semi_pos);
        arglist.substring(pos, 2 + semi_pos);
    }
}
