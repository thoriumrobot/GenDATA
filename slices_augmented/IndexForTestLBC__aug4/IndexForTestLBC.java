/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
package index;

import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForTestLBC {

    int[] array;

    void test1(@IndexFor("array") int i) {
        Float __cfwr_temp74 = null;

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
    public static boolean __cfwr_process375(float __cfwr_p0, String __cfwr_p1) {
        try {
            try {
            short __cfwr_val25 = ((-2.15 * true) - 18.24f);
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        while (false) {
            try {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 6; __cfwr_i42++) {
            try {
            while (true) {
            if (true || true) {
            String __cfwr_obj34 = "value13";
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        try {
            if ((642L | null) || true) {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 1; __cfwr_i15++) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 4; __cfwr_i6++) {
            return 815;
        }
        }
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        return (-72.33 - null);
    }
    protected static Boolean __cfwr_util570(String __cfwr_p0, Double __cfwr_p1) {
        if (true || true) {
            return 484;
        }
        return null;
        Double __cfwr_result71 = null;
        return null;
    }
}
