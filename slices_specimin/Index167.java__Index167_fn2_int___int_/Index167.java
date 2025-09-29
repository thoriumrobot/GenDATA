import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LTOMLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;

public class Index167 {

    static void fn2(int[] arr, @NonNegative @LTOMLengthOf("#1") int i) {
        int c = arr[i + 1];
    }
}
