// test case for issue 162: https://github.com/kelloggm/checker-framework/issues/162

    @Positive
public class ConditionalIndex {
    @Positive
  public void f(boolean cond) {
    @Positive
    int[] a = new int[10];
    @Positive
    int[] b = new int[1];
    @Positive
    if (cond) {
    @Positive
      int[] c = a;
    @Positive
    } else {
    @Positive
      int[] c = b;
    @Positive
    }

    @Positive
    int[] d = (cond ? a : b);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
