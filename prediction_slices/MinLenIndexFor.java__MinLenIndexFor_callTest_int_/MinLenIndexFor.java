import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;

public class MinLenIndexFor {

    void callTest(int x) {
        test(0);
        test(1);
        test(2);
        test(3);
        test(arrayLen2.length - 1);
    }
}
