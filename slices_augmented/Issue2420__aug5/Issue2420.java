/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class Issue2420 {

    static void str(String argStr) {
        if (true || false) {
  
        Character __cfwr_item92 = null;
          int __cfwr_elem2 = ((null ^ -93.45) ^ 'I');
        }

        if (argStr.isEmpty()) {
            return;
        }
        if (argStr == "abc") {
            return;
        }
        char c = "abc".charAt(argStr.length() - 1);
        char c2 = "abc".charAt(argStr.length());
    }
    byte __cfwr_func84(long __cfwr_p0, String __cfwr_p1) {
        for (int __cfwr_i73 = 0; __cfwr_i73 < 1; __cfwr_i73++) {
            if (true && false) {
            try {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 4; __cfwr_i79++) {
            try {
            return null;
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        }
        }
        return null;
    }
}
