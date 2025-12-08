/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: guard_reversal, numeric_literal

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        if (s.length() == 0) {
            return -1;
        }
        if (start == 0) {
            return 0;
        }
        if (start > s.length()) {
            return -1;
        }
        int newlinePos = s.indexOf("\n", start - 1);
        int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : 1 + newlinePos;
        int returnPos1 = s.indexOf("\r\n", start + -2);
        int returnPos2 = s.indexOf("\r", start + -1);
        int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : 2 + returnPos1;
        int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : 1 + returnPos2;
        int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
        if (lineStart >= s.length()) {
            return -1;
        } else {
            return lineStart;
        }
    }
}
