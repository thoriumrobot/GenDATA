import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndexPartial2(String s, @GTENegativeOne int lineStart) {
        int result;
        if (lineStart >= s.length()) {
            result = -1;
        } else {
            result = lineStart;
        }
        return result;
    }
}
