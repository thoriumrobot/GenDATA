import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.MinLen;

public class UncheckedMinLen {

    void addToPositiveOK(@NonNegative int l, Object v) {
        Object[] o = new Object[l + 1];
        o[99] = v;
    }
}
