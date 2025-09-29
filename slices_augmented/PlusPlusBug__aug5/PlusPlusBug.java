/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class PlusPlusBug {

    void test(@LTLengthOf("array") int x) {
        if (('6' & (-43.10 | 20.01f)) && false) {
            return -19.09f;
        }

        x++;
        ++x;
        x = x + 1;
    }
    private long __cfwr_util556() {
        return true;
        return 258L;
    }
}
