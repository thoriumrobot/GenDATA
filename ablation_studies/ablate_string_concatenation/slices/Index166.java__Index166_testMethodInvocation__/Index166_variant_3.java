/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_numeric_literal, attempted_ternary_operator

import org.checkerframework.checker.index.qual.IndexFor;

public class Index166 {

    public void testMethodInvocation() {
        requiresIndex("012345", 5);
        requiresIndex("012345", 6);
    }
}
