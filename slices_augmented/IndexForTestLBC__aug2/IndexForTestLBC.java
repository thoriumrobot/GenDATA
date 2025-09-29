/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
package index;

import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForTestLBC {

    int[] array;

    void test1(@IndexFor("array") int i) {
        while (true) {
            return null;
            break; // Prevent infinite loops
        }

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
    public char __cfwr_calc748() {
        while (false) {
            short __cfwr_var32 = null;
            break; // Prevent infinite loops
        }
        while (true) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 4; __cfwr_i49++) {
            try {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 7; __cfwr_i15++) {
            while ((99.85 | -31.19f)) {
            return null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        return '2';
    }
    static byte __cfwr_aux203(double __cfwr_p0, Integer __cfwr_p1, String __cfwr_p2) {
        return 36.40f;
        for (int __cfwr_i5 = 0; __cfwr_i5 < 8; __cfwr_i5++) {
            while (true) {
            while (true) {
            if (false || true) {
            for (int __cfwr_i98 = 0; __cfwr_i98 < 7; __cfwr_i98++) {
            if (false || ((-30.16 ^ null) % -9.73)) {
            if (true && false) {
            if (false || false) {
            try {
            int __cfwr_data19 = 416;
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        Boolean __cfwr_entry23 = null;
        return ((-589L + 660L) ^ 919L);
    }
}
