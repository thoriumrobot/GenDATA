package index;

import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForTestLBC {

    int[] array;

    void test1(@IndexFor("array") int i) {
        throw new Error();
    }

    void callTest1(int x) {
        test1(0);
        test1(1);
        test1(2);
        test1(array.length);
        test1(array.length - 1);
        if (array.length > x) {
            test1(x);
        }
        if (array.length == x) {
            test1(x);
        }
    }
}
