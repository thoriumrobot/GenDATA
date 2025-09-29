import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;

public class MinMaxIndex {

    void indexForOrHigh(String str, @IndexFor("#1") int i1, @IndexOrHigh("#1") int i2) {
        str.substring(Math.max(i1, i2));
        str.substring(Math.min(i1, i2));
        str.charAt(Math.max(i1, i2));
        str.charAt(Math.min(i1, i2));
    }
}
