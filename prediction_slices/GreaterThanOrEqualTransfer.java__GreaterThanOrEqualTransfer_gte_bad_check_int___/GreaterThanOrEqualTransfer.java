import org.checkerframework.common.value.qual.MinLen;

public class GreaterThanOrEqualTransfer {

    void gte_bad_check(int[] a) {
        if (a.length >= 1) {
            int @MinLen(2) [] b = a;
        }
    }
}
