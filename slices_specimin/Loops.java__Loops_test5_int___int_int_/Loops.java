import org.checkerframework.checker.index.qual.LTLengthOf;

public class Loops {

    public void test5(int[] a, @LTLengthOf(value = "#1", offset = "-1000") int offset, @LTLengthOf("#1") int offset2) {
        int otherOffset = offset;
        while (flag) {
            otherOffset += 1;
            offset++;
            offset += 1;
            offset2 += offset;
        }
        @LTLengthOf(value = "#1", offset = "-1000")
        int x = otherOffset;
    }
}
