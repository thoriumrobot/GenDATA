import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineGT {

    void testLTL(@LTLengthOf("arr") int test) {
        @LTLengthOf("arr")
        int a = Integer.parseInt("1");
        @LTLengthOf("arr")
        int a3 = Integer.parseInt("3");
        int b = 2;
        if (test > b) {
            @LTLengthOf("arr")
            int c = b;
        }
        @LTLengthOf("arr")
        int c1 = b;
        if (a > b) {
            int potato = 7;
        } else {
            @LTLengthOf("arr")
            int d = b;
        }
    }
}
