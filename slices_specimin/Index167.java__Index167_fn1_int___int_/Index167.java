import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LTOMLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;

public class Index167 {

    static void fn1(int[] arr, @IndexFor("#1") int i) {
        if (i >= 33) {
            fn2(arr, i);
        }
        if (i > 33) {
            fn2(arr, i);
        }
        if (i != 33) {
            fn2(arr, i);
        }
    }
}
