import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class IndexOrLowTests {

    void test() {
        if (index != -1) {
            array[index] = 1;
        }
        @IndexOrHigh("array")
        int y = index + 1;
        array[y] = 1;
        if (y < array.length) {
            array[y] = 1;
        }
        index = array.length;
    }
}
