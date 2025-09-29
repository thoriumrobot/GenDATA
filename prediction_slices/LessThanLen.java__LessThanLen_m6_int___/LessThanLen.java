import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.MinLen;

public class LessThanLen {

    public static void m6(int @MinLen(1) [] shorter) {
        int[] longer = new int[4 * shorter.length];
        @LTEqLengthOf("longer")
        int y = shorter.length;
    }
}
