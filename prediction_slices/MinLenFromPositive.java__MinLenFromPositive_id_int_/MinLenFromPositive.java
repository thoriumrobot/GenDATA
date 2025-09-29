import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.*;

public class MinLenFromPositive {

    @Positive
    int id(@Positive int x) {
        return x;
    }
}
