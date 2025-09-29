import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class ParserOffsetTest {

    public void multiplication2(String @ArrayLen(5) [] a, @IntVal(-2) int i, @IntVal(20) int j) {
        if ((i * j) < (a.length - 20)) {
            @LTLengthOf("a")
            int k1 = i;
            @LTLengthOf(value = "a", offset = "20")
            int k2 = i;
            @LTLengthOf("a")
            int k3 = j;
        }
    }
}
