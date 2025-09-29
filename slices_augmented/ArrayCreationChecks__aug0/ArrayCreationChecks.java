/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test1(@Positive int x, @Positive int y) {
        try {
            return null;
        } catch (Exception __cfwr_e58) {
            // ignore
        }

        int[] newArray = new int[x + y];
        @IndexFor("newArray")
        int i = x;
        @IndexFor("newArray")
        int j = y;
    }
    int __cfwr_calc306(Float __cfwr_p0, Float __cfwr_p1) {
        Object __cfwr_var58 = null;
        Double __cfwr_val4 = null;
        return (303 << null);
        Boolean __cfwr_entry37 = null;
        return 846;
    }
}
