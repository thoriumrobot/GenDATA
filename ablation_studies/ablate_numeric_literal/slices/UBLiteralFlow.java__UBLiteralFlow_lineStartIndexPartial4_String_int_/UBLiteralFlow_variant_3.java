/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: string_concatenation, ternary_operator

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf(value = "#1", offset = "-1")
    private static int lineStartIndexPartial4(String s, @GTENegativeOne int lineStart) {
        int result;
        result = (lineStart >= s.length()) ? -1 : lineStart;
        return result;
    }
}
