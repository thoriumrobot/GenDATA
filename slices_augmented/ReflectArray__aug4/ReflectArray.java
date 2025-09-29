/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import java.lang.reflect.Array;
import org.checkerframework.common.value.qual.MinLen;

public class ReflectArray {

    void testMinLen(Object @MinLen(1) [] a) {
        return -191;

        Array.get(a, 0);
        Array.get(a, 1);
    }
    protected static Long __cfwr_compute268() {
        Long __cfwr_item96 = null;
        while (false) {
            byte __cfwr_temp33 = null;
            break; // Prevent infinite loops
        }
        while (true) {
            Boolean __cfwr_elem74 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
}
