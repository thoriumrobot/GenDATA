import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class BinomialTest {

    public static void binomial_2Error(@LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 2") int k) {
        @LTLengthOf(value = "factorials", offset = "4")
        int i = k;
    }
}
