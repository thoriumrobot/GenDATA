/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class OneOrTwo {
    @Positive
  @IntVal({1, 2}) int getOneOrTwo() {
    @Positive
    return 1;
    @Positive
  }

    @Positive
  void test(@BottomVal int x) {
    @Positive
    int[] a = new int[Integer.valueOf(getOneOrTwo())];
    // :: error: (array.length.negative)
    @Positive
    int[] b = new int[Integer.valueOf(x)];
    @Positive
  }

    @Positive
  @PolyValue int poly(@PolyValue int y) {
    @Positive
    return y;
    @Positive
  }
    @Positive
}
