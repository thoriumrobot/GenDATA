import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineGTE {

    void testLTEL(@LTEqLengthOf("arr") int test) {
        @LTEqLengthOf("arr")
        int a = Integer.parseInt("1");
        @LTEqLengthOf("arr")
        int a3 = Integer.parseInt("3");
        int b = 2;
        if (test >= b) {
            @LTEqLengthOf("arr")
            int c = b;
        }
        @LTEqLengthOf("arr")
        int c1 = b;
        if (a >= b) {
            int potato = 7;
        } else {
            @LTEqLengthOf("arr")
            int d = b;
        }
    }
}
