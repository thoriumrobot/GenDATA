import org.checkerframework.checker.index.qual.LTLengthOf;

public class Loops {

    public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        while (flag) {
            offset++;
        }
    }
}
