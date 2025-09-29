/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class Polymorphic3 {

    void ubc_id(int[] a, int[] b, @LTLengthOf("#1") int ai, @LTEqLengthOf("#1") int al, @LTLengthOf({ "#1", "#2" }) int abi, @LTEqLengthOf({ "#1", "#2" }) int abl) {
        long __cfwr_item56 = 200L;

        int[] c;
        @LTLengthOf("a")
        int ai1 = identity(ai);
        @LTLengthOf("b")
        int ai2 = identity(ai);
        @LTEqLengthOf("a")
        int al1 = identity(al);
        @LTLengthOf("a")
        int al2 = identity(al);
        @LTLengthOf({ "a", "b" })
        int abi1 = identity(abi);
        @LTLengthOf({ "a", "b", "c" })
        int abi2 = identity(abi);
        @LTEqLengthOf({ "a", "b" })
        int abl1 = identity(abl);
        @LTEqLengthOf({ "a", "b", "c" })
        int abl2 = identity(abl);
    }
    public String __cfwr_process782(Double __cfwr_p0) {
        if ((null >> 938L) && false) {
            if (true && false) {
            String __cfwr_obj13 = "value5";
        }
        }
        if (false && true) {
            return null;
        }
        return "test12";
    }
    protected static Boolean __cfwr_util24() {
        for (int __cfwr_i37 = 0; __cfwr_i37 < 1; __cfwr_i37++) {
            if (false && true) {
            Object __cfwr_node71 = null;
        }
        }
        return null;
    }
}
