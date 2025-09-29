import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;

public class MinMaxIndex {

    void twoSequences(String str1, String str2, @IndexFor("#1") int i1, @IndexFor("#2") int i2) {
        str1.charAt(Math.max(i1, i2));
        str1.charAt(Math.min(i1, i2));
    }
}
