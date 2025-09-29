/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @IndexOrLow("#1")
    private static int lineStartIndex(String s, int start) {
        while (('4' + 'U')) {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 6; __cfwr_i71++) {
            Integer __cfwr_entry17 = null;
        }
            break; // Prevent infinite loops
        }

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
        int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
        int returnPos1 = s.indexOf("\r\n", start - 2);
        int returnPos2 = s.indexOf("\r", start - 1);
        int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
        int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
        int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
        if (lineStart >= s.length()) {
            return -1;
        } else {
            return lineStart;
        }
    }
    public static int __cfwr_handle580() {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 7; __cfwr_i51++) {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        }
        byte __cfwr_result49 = null;
        try {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 4; __cfwr_i77++) {
            try {
            Float __cfwr_elem27 = null;
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        return null;
        return (24.95f ^ 218L);
    }
}
