package lessthan;

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class LessThanValue {

    @NonNegative
    int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
        throw new Error();
    }

    void test(int maximum, int count) {
        if (maximum < 0) {
            throw new IllegalArgumentException("Negative 'maximum' argument.");
        }
        if (count > maximum) {
            int deleteIndex = count - maximum - 1;
            isLessThanOrEqual(0, deleteIndex);
        }
    }
}
