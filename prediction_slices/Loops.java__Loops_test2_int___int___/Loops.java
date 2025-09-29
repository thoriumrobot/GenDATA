import org.checkerframework.checker.index.qual.LTLengthOf;

public class Loops {

    public void test2(int[] a, int[] array) {
        int offset = array.length - 1;
        int offset2 = array.length - 1;
        while (flag) {
            offset++;
            offset2 += offset;
        }
        @LTLengthOf("array")
        int x = offset;
        @LTLengthOf("array")
        int y = offset2;
    }
}
