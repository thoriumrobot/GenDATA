import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test4(@GTENegativeOne int x, @NonNegative int y) {
        int[] newArray = new int[x + y];
        @LTEqLengthOf("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
}
