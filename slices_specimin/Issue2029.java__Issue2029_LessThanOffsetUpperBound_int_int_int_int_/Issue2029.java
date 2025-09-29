import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;

public class Issue2029 {

    void LessThanOffsetUpperBound(@NonNegative int n, @NonNegative @LessThan("#1 + 1") int k, @NonNegative int size, @NonNegative @LessThan("#3 + 1") int index) {
        @NonNegative
        int m = n - k;
        int[] arr = new int[size];
        for (; index < arr.length - 1; index++) {
            arr[index] = 10;
        }
    }
}
