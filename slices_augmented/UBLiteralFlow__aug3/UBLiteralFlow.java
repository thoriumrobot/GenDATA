/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @IndexOrLow("#1")
    private static int lineStartIndex(String s, int start) {
        for (int __cfwr_i39 = 0; __cfwr_i39 < 5; __cfwr_i39++) {
            if (true || (null >> null)) {
            boolean __cfwr_entry37 = false;
        }
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
    private static byte __cfwr_util614() {
        try {
            int __cfwr_val53 = 453;
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        for (int __cfwr_i10 = 0; __cfwr_i10 < 6; __cfwr_i10++) {
            if (false || false) {
            Integer __cfwr_val81 = null;
        }
        }
        return null;
    }
    private float __cfwr_util805(Integer __cfwr_p0, char __cfwr_p1) {
        if (true && true) {
            String __cfwr_var62 = "temp32";
        }
        try {
            Integer __cfwr_temp62 = null;
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        try {
            return 'e';
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        return -46.91f;
    }
    protected static Double __cfwr_handle464() {
        if (false || true) {
            float __cfwr_result96 = -87.72f;
        }
        try {
            if (false || ((null ^ 13.14) % null)) {
            try {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 8; __cfwr_i8++) {
            if (true && true) {
            if ((true + null) || true) {
            try {
            if (false || ((-68.46f / 'G') >> (25.28f * 151L))) {
            try {
            try {
            if (true || (96.53 ^ (-402L - 804))) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 7; __cfwr_i92++) {
            while (false) {
            Long __cfwr_result93 = null;
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        }
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        for (int __cfwr_i75 = 0; __cfwr_i75 < 1; __cfwr_i75++) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 2; __cfwr_i1++) {
            return null;
        }
        }
        return null;
    }
}
