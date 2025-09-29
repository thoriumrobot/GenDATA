/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
package index;

import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForTestLBC {

    int[] array;

    void test1(@IndexFor("array") int i) {
        return -985L;

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
    private static Character __cfwr_func995() {
        String __cfwr_temp97 = "result89";
        char __cfwr_node97 = (555L << (-43.52f % true));
        Long __cfwr_val97 = null;
        for (int __cfwr_i78 = 0; __cfwr_i78 < 10; __cfwr_i78++) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    protected Integer __cfwr_process370(Double __cfwr_p0, double __cfwr_p1) {
        while (true) {
            while ((null - false)) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
