package lessthan;

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class LessThanValue {

    void isLessThan(@LessThan("#2") @NonNegative int start, int end) {
        throw new Error();
    }

    @NonNegative
    int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
        throw new Error();
    }

    void count(int count) {
        if (count > 0) {
            if (count % 2 == 1) {
            } else {
                int countDivMinus = count / 2 - 1;
                countDivMinus = countDivMinus;
                isLessThan(0, countDivMinus);
                isLessThanOrEqual(0, countDivMinus);
            }
        }
    }
}
