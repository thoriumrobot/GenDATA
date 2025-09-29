package lessthan;

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class LessThanValue {

    @NonNegative
    @LessThan("#2 + 1")
    static int expandedCapacity(@NonNegative int oldCapacity, @NonNegative int minCapacity) {
        if (minCapacity < 0) {
            throw new AssertionError("cannot store more than MAX_VALUE elements");
        }
        int newCapacity = oldCapacity + (oldCapacity >> 1) + 1;
        if (newCapacity < minCapacity) {
            newCapacity = Integer.highestOneBit(minCapacity - 1) << 1;
        }
        if (newCapacity < 0) {
            newCapacity = Integer.MAX_VALUE;
        }
        return newCapacity;
    }
}
