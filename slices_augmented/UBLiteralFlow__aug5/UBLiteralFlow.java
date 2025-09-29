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
            try 
        Character __cfwr_data95 = null;
{
            try {
            if (false && false) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 9; __cfwr_i32++) {
            try {
            char __cfwr_temp65 = 'V';
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        } catch (Exception __cfwr_e10) {
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
    protected long __cfwr_proc440(float __cfwr_p0, byte __cfwr_p1, double __cfwr_p2) {
        Long __cfwr_val53 = null;
        return 952L;
    }
    short __cfwr_process989(String __cfwr_p0, Integer __cfwr_p1, Object __cfwr_p2) {
        byte __cfwr_temp33 = (146L * null);
        return false;
        for (int __cfwr_i46 = 0; __cfwr_i46 < 4; __cfwr_i46++) {
            try {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        }
        if (false && true) {
            try {
            return (null % false);
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        return null;
    }
    public static Double __cfwr_aux515(long __cfwr_p0) {
        int __cfwr_val59 = (null | '9');
        while (false) {
            try {
            for (int __cfwr_i72 = 0; __cfwr_i72 < 3; __cfwr_i72++) {
            while ((false >> (980 & null))) {
            return (false % (-47.42f << null));
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
