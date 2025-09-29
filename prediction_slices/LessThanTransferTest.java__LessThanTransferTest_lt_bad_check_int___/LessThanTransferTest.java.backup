import org.checkerframework.common.value.qual.MinLen;

public class LessThanTransferTest {

    void lt_bad_check(int[] a) {
        if (0 < a.length) {
            int @MinLen(2) [] b = a;
        }
    }
}
