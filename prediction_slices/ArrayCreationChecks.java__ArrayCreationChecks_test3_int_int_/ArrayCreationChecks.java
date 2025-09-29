import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test3(@NonNegative int x, @NonNegative int y) {
        int[] newArray = new int[x + y];
        @IndexOrHigh("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
}
