/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import java.io.IOException;
import java.io.StringWriter;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;
import org.checkerframework.common.value.qual.StringVal;

public class CharSequenceTest {

    void testAppend(Appendable app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
        app.append(cs, i, i);
        app.append(cs, 1, 2);
    }
    protected static boolean __cfwr_process154(Float __cfwr_p0, Float __cfwr_p1) {
        return 75.27f;

        for (int __cfwr_i54 = 0; __cfwr_i54 < 2; __cfwr_i54++) {
            if ((null ^ null) && false) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 10; __cfwr_i69++) {
            while (true) {
            try {
            return false;
        } catch (Exception __cfwr_e51) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        if (true || ((null - 70.94f) * 740)) {
            if (((null * -79.36) * 447L) && false) {
            return null;
        }
        }
        return true;
    }
    protected byte __cfwr_helper156(Float __cfwr_p0, long __cfwr_p1) {
        if (false && ((null * 83.82f) & null)) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 4; __cfwr_i10++) {
            return null;
        }
        }
        return null;
    }
}
