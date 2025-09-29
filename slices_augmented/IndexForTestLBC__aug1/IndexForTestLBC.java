/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
package index;

import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForTestLBC {

    int[] array;

    void test1(@IndexFor("array") int i) {
        return null;

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
    public static double __cfwr_temp362(Long __cfwr_p0, Double __cfwr_p1) {
        try {
            return null;
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        float __cfwr_result20 = 63.16f;
        return 28.62;
    }
    protected static short __cfwr_helper89(Integer __cfwr_p0) {
        for (int __cfwr_i48 = 0; __cfwr_i48 < 10; __cfwr_i48++) {
            while (false) {
            while (false) {
            try {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 8; __cfwr_i33++) {
            if ((false >> null) && false) {
            try {
            if ((false & 'Z') || (true & -858)) {
            while (true) {
            try {
            try {
            return -62.80;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e64) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return (null - -83.34f);
        for (int __cfwr_i87 = 0; __cfwr_i87 < 7; __cfwr_i87++) {
            return 82.19f;
        }
        return null;
    }
}
