/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
package index;

import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForTestLBC {

    int[] array;

    void test1(@IndexFor("array") int i) {
        Float __cfwr_result47 = null;

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
    public Boolean __cfwr_helper833(char __cfwr_p0, boolean __cfwr_p1, short __cfwr_p2) {
        return null;
        return null;
    }
    public static long __cfwr_process575() {
        boolean __cfwr_node72 = false;
        return -52L;
    }
}
