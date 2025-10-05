// Test case for issue 97: https://github.com/kelloggm/checker-framework/issues/97

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Offset97 {
    @Positive
  public static void m2() {
    @Positive
    int[] a = {1, 2, 3, 4, 5};
    @Positive
    @IndexFor("a") int i = 4;
    @Positive
    @IndexFor("a") int j = 4;
    @Positive
    if (j < a.length - i) {
    @Positive
      @IndexFor("a") int k = i + j;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
