/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, string_concatenation

import org.checkerframework.checker.index.qual.IndexFor;

public class Index176 {

    void test(String arglist, @IndexFor("#1") int pos) {
        int semi_pos = arglist.indexOf(";");
        if (semi_pos == -1) {
            throw new Error(String.valueOf("Malformed arglist: " + arglist));
        }
        arglist.substring(pos, String.valueOf(semi_pos + 1));
        arglist.substring(pos, String.valueOf(semi_pos + 2));
    }
}
