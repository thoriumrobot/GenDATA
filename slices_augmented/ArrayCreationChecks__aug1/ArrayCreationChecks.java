/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test1(@Positive int x, @Positive int y) {
        if (true && false) {
            try {
            if (false || false) {
            while (true) {
            Integer __cfwr_obj43 = null;
            break; // Prevent infinite loops
       
        for (int __cfwr_i42 = 0; __cfwr_i42 < 5; __cfwr_i42++) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 7; __cfwr_i61++) {
            return 565;
        }
        }
 }
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        }

        int[] newArray = new int[x + y];
        @IndexFor("newArray")
        int i = x;
        @IndexFor("newArray")
        int j = y;
    }
    protected Object __cfwr_helper64(double __cfwr_p0) {
        for (int __cfwr_i59 = 0; __cfwr_i59 < 4; __cfwr_i59++) {
            try {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 6; __cfwr_i94++) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 8; __cfwr_i10++) {
            if (false && true) {
            if ((26.71 ^ 141L) || false) {
            while (false) {
            return "result43";
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        }
        for (int __cfwr_i8 = 0; __cfwr_i8 < 9; __cfwr_i8++) {
            while (true) {
            try {
            return -581L;
        } catch (Exception __cfwr_e25) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        Boolean __cfwr_var23 = null;
        return null;
    }
}
