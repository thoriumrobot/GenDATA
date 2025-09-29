import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;

public class MinLenIndexFor {

    void callTest2(int x) {
        test2(0);
        test2(1);
        test2(2);
        test2(4);
        test2(5);
        test2(arrayLen4.length);
    }
}
