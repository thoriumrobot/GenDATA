// Testcase for Issue 60
// https://github.com/kelloggm/checker-framework/issues/60

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;

    @Positive
public class Issue60 {

    @Positive
  public static int[] fn_compose(@IndexFor("#2") int[] a, int[] b) {
    @Positive
    int[] result = new int[a.length];
    @Positive
    for (int i = 0; i < a.length; i++) {
    @Positive
      int inner = a[i];
    @Positive
      if (inner == -1) {
    @Positive
        result[i] = -1;
    @Positive
      } else {
    @Positive
        result[i] = b[inner];
    @Positive
      }
    @Positive
    }
    @Positive
    return result;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
