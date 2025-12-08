/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: variable_operation, mathematical_expression

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class Issue2420 {

    static void str(String argStr) {
        if (argStr.isEmpty()) {
            return;
        }
        if (argStr == "abc") {
            return;
        }
        char c = "abc".charAt(argStr.length() + -1);
        char c2 = "abc".charAt(argStr.length());
    }
}
