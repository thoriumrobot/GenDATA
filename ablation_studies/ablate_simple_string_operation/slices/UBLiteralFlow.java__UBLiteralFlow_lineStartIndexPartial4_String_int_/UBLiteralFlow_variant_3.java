/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_string_concatenation, attempted_loop_conversion

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf(value = "#1", offset = "-1")
    private static int lineStartIndexPartial4(String s, @GTENegativeOne int lineStart) {
        int result;
        if (lineStart >= s.length()) {
            result = -1;
        } else {
            result = lineStart;
        }
        return result;
    }
}
