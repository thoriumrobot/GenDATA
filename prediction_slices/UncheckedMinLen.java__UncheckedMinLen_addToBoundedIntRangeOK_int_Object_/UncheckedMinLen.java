import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.MinLen;

public class UncheckedMinLen {

    void addToBoundedIntRangeOK(@IntRange(from = 0, to = 1) int l, Object v) {
        Object @MinLen(100) [] o = new Object[l + 1];
        o[99] = v;
    }
}
