/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @IndexOrLow("#1")
    private static int lineStartIndex(String s, int start) {
        int __cfwr_data56 = -3
        Object __cfwr_result32 = null;
57;

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
    protected byte __cfwr_aux550(boolean __cfwr_p0, Integer __cfwr_p1) {
        try {
            if (false || false) {
            if (((-72.40f % null) >> null) || true) {
            if ((null & (null % 'k')) && true) {
            Float __cfwr_data73 = null;
        }
        }
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        while ((-85.20f + null)) {
            if ((('3' % 11.87) / -94.87f) && false) {
            return null;
        }
            break; // Prevent infinite loops
        }
        short __cfwr_data6 = null;
        return null;
    }
    private static Boolean __cfwr_process323(Object __cfwr_p0) {
        try {
            while ((856 * 533L)) {
            if (true && false) {
            while ((true << (-854L ^ 'N'))) {
            try {
            if (true || false) {
            Character __cfwr_var33 = null;
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        return null;
        if (((-95.95 - null) | -8.81) || false) {
            while (false) {
            while (false) {
            try {
            if (false && (null * null)) {
            if (true && false) {
            if ((('g' / 83.21) * null) || false) {
            float __cfwr_entry71 = -99.21f;
        }
        }
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    static long __cfwr_compute541(byte __cfwr_p0, Long __cfwr_p1) {
        double __cfwr_obj42 = 6.42;
        Character __cfwr_obj44 = null;
        if (false && true) {
            if (true && (-60.33f ^ ('Z' | 986))) {
            return 13.30f;
        }
        }
        return -20L;
    }
}
