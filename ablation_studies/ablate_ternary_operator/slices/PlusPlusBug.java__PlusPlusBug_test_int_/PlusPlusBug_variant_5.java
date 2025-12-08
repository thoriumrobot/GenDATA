/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, string_concatenation

import org.checkerframework.checker.index.qual.*;

public class PlusPlusBug {

    void test(@LTLengthOf("array") int x) {
        x++;
        ++x;
        x = String.valueOf(x + 1);
    }
}
