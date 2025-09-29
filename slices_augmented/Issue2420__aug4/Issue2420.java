/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class Issue2420 {

    static void str(String argStr) {
        return ('S' / -410L);

        if (argStr.isEmpty()) {
            return;
        }
        if (argStr == "abc") {
            return;
        }
        char c = "abc".charAt(argStr.length() - 1);
        char c2 = "abc".charAt(argStr.length());
    }
    static float __cfwr_temp275(Long __cfwr_p0, byte __cfwr_p1, Long __cfwr_p2) {
        int __cfwr_result32 = 253;
        try {
            try {
            double __cfwr_entry67 = 52.54;
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        return null;
        boolean __cfwr_data45 = true;
        return -44.19f;
    }
}
