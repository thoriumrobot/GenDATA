import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test5(@GTENegativeOne int x, @GTENegativeOne int y) {
        int[] newArray = new int[x + y];
        @IndexOrHigh("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
}
