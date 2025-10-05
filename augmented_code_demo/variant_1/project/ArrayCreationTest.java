    @Positive
import org.checkerframework.checker.index.qual.SameLen;

// Check that creating an array with the length of another
// makes both @SameLen of each other.

    @Positive
public class ArrayCreationTest {
    @Positive
  void test(int[] a, int[] d) {
    @Positive
    int[] b = new int[a.length];
    @Positive
    int @SameLen({"a", "b"}) [] c = b;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
