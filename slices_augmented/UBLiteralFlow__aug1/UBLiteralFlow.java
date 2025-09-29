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
            whil
        try {
            if (true || true) {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 6; __cfwr_i66++) {
            if (true && (true >> -701)) {
            Float __cfwr_data89 = null;
        }
        }
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }
e (true) {
            if (true && false) {
            try {
            if (true && (-858L - 'n')) {
            try {
            Character __cfwr_item12 = null;
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e85) {
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
    private static char __cfwr_process11(Object __cfwr_p0, byte __cfwr_p1) {
        Double __cfwr_result52 = null;
        while (((-58.95f - null) << true)) {
            float __cfwr_item9 = (null >> (null * null));
            break; // Prevent infinite loops
        }
        return 'N';
    }
    static Boolean __cfwr_func783(Object __cfwr_p0, long __cfwr_p1) {
        try {
            int __cfwr_val31 = 470;
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        char __cfwr_result66 = 'C';
        return null;
    }
    static Long __cfwr_util884(Integer __cfwr_p0, float __cfwr_p1, char __cfwr_p2) {
        return null;
        if (true || false) {
            while (((null >> null) << 477L)) {
            try {
            return "hello86";
        } catch (Exception __cfwr_e56) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}
