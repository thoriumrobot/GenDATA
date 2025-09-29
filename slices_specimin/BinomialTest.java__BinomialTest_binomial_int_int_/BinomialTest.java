import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class BinomialTest {

    public static long binomial(@NonNegative @LTLengthOf("BinomialTest.factorials") int n, @NonNegative @LessThan("#1 + 1") int k) {
        return factorials[k];
    }
}
