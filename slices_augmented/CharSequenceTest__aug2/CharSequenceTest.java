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
    protected int __cfwr_util497(boolean __cfwr_p0) {
        Boolean __cfwr_obj25 = null;

        try {
            Boolean __cfwr_item79 = null;
  
        return null;
      } catch (Exception __cfwr_e68) {
            // ignore
        }
        while (true) {
            try {
            if (true || true) {
            if (false && (40.45 << (false ^ 14.11f))) {
            if (false || true) {
            try {
            if (true || true) {
            return null;
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        }
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return 204;
    }
}
