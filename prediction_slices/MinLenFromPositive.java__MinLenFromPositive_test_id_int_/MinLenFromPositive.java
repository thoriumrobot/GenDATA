import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.*;

public class MinLenFromPositive {

    void test_id(int param) {
        @Positive
        int x = id(5);
        @IntRange(from = 1)
        int y = id(5);
        int @MinLen(1) [] a = new int[id(100)];
        int @MinLen(10) [] c = new int[id(100)];
        int q = id(10);
        if (param == q) {
            int @MinLen(1) [] d = new int[param];
        }
    }
}
