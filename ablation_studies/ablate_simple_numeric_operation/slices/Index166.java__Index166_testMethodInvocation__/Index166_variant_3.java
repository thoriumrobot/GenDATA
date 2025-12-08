/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_string_concatenation, attempted_loop_conversion

import org.checkerframework.checker.index.qual.IndexFor;

public class Index166 {

    public void testMethodInvocation() {
        requiresIndex("012345", 5);
        requiresIndex("012345", 6);
    }
}
