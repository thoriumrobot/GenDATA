/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class UpperBoundRefinement {
  // If expression i has type @LTLengthOf(value = "f2", offset = "f1.length") int and expression
  // j is less than or equal to the length of f1, then the type of i + j is @LTLengthOf("f2")
    @Positive
  void test(int[] f1, int[] f2) {
    @Positive
    @LTLengthOf(value = "f2", offset = "f1.length") int i = (f2.length - 1) - f1.length;
    @Positive
    @LTLengthOf("f1") int j = f1.length - 1;
    @Positive
    @LTLengthOf("f2") int x = i + j;
    @Positive
    @LTLengthOf("f2") int y = i + f1.length;
    @Positive
  }

    @Positive
  void test2() {
    @Positive
    double[] f1 = new double[10];
    @Positive
    double[] f2 = new double[20];

    @Positive
    for (int j = 0; j < f2.length; j++) {
    @Positive
      f2[j] = j;
    @Positive
    }
    @Positive
    for (int i = 0; i < f2.length - f1.length; i++) {
      // fill up f1 with elements of f2
    @Positive
      for (int j = 0; j < f1.length; j++) {
    @Positive
        f1[j] = f2[i + j];
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  public void test3(double[] a, double[] sub) {
    @Positive
    int a_index_max = a.length - sub.length;
    // Has type @LTL(value={"a","sub"}, offset={"-1 + sub.length", "-1 + a.length"})

    @Positive
    for (int i = 0; i <= a_index_max; i++) { // i has the same type as a_index_max
    @Positive
      for (int j = 0; j < sub.length; j++) { // j is @LTL("sub")
        // i + j is safe here.
        // Because j is LTL("sub"), it should count as ("-1 + sub.length")
    @Positive
        double d = a[i + j];
    @Positive
      }
    @Positive
    }
    @Positive
  }
    @Positive
}
