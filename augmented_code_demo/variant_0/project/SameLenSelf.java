// Test case for issue 146: https://github.com/kelloggm/checker-framework/issues/146

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SameLenSelf {
    @Positive
  int @SameLen("this.field") [] field = new int[10];
    @Positive
  int @SameLen("field2") [] field2 = new int[10];
    @Positive
  int @SameLen("field3") [] field3 = field2;

    @Positive
  void foo(int[] b) {
    @Positive
    int @SameLen("a") [] a = b;
    @Positive
    int @SameLen("c") [] c = new int[10];
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
