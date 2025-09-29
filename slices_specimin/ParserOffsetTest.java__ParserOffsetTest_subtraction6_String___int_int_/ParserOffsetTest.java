import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class ParserOffsetTest {

    public void subtraction6(String[] a, int i, int j) {
        if (i - j < a.length - 1) {
            @IndexFor("a")
            int k = i - j;
            @IndexFor("a")
            int k1 = i;
        }
    }
}
