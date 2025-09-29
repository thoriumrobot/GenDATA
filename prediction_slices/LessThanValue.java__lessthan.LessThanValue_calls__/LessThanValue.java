package lessthan;

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class LessThanValue {

    void calls() {
        isLessThan(0, 1);
        isLessThanOrEqual(0, 0);
    }

    void isLessThan(@LessThan("#2") @NonNegative int start, int end) {
        throw new Error();
    }

    @NonNegative
    int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
        throw new Error();
    }
}
