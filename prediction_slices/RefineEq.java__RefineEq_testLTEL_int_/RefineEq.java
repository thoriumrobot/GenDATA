import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineEq {

    void testLTEL(@LTEqLengthOf("arr") int test) {
        @LTEqLengthOf("arr")
        int a = Integer.parseInt("1");
        int b = 1;
        if (test == b) {
            @LTEqLengthOf("arr")
            int c = b;
            @LTLengthOf("arr")
            int g = b;
        } else {
            @LTEqLengthOf("arr")
            int e = b;
        }
        @LTEqLengthOf("arr")
        int d = b;
    }
}
