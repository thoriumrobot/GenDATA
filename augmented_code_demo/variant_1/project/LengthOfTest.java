    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class LengthOfTest {
    @Positive
  void foo(int[] a, @LengthOf("#1") int x) {
    @Positive
    @IndexOrHigh("a") int y = x;
    // :: error: (assignment)
    @Positive
    @IndexFor("a") int w = x;
    @Positive
    @LengthOf("a") int z = a.length;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
