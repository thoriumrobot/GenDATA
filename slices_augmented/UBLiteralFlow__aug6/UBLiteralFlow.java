/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @IndexOrLow("#1")
    private static int lineStartIndex(String s, int start) {
        try {
            for 
        if (false && true) {
            try {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 2; __cfwr_i3++) {
            return null;
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        }
(int __cfwr_i88 = 0; __cfwr_i88 < 8; __cfwr_i88++) {
            while (false) {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 4; __cfwr_i35++) {
            while (false) {
            for (int __cfwr_i38 = 0; __cfwr_i38 < 8; __cfwr_i38++) {
            String __cfwr_result30 = "item10";
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
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
    private static Boolean __cfwr_proc103(Boolean __cfwr_p0, String __cfwr_p1, float __cfwr_p2) {
        for (int __cfwr_i84 = 0; __cfwr_i84 < 8; __cfwr_i84++) {
            try {
            if (true && ((90.51f << null) % 984)) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 7; __cfwr_i1++) {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 9; __cfwr_i82++) {
            return 779;
        }
        }
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        }
        return null;
        if (true || (true & 702L)) {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 5; __cfwr_i52++) {
            for (int __cfwr_i20 = 0; __cfwr_i20 < 3; __cfwr_i20++) {
            return -833;
        }
        }
        }
        return null;
    }
}
