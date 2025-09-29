import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.common.value.qual.MinLen;

public class IndexForTest {

    void callTest2(int x) {
        test2(0);
        test2(1);
        test2(2);
        test2(array.length);
        if (array.length > 0) {
            test2(array.length - 1);
        }
        test2(array.length - 1);
        test2(this.array.length);
        if (array.length > 0) {
            test2(this.array.length - 1);
        }
        test2(this.array.length - 1);
        if (array.length == x && x >= 0) {
            test2(x);
        }
    }
}
