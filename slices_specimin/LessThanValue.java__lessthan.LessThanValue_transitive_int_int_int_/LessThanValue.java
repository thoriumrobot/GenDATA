package lessthan;

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class LessThanValue {

    void transitive(int a, int b, int c) {
        if (a < b) {
            if (b < c) {
                @LessThan("c")
                int x = a;
            }
        }
    }
}
