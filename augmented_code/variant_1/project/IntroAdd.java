/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class IntroAdd {
    @Positive
  void test(int[] arr) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a = 3;
    @Positive
  }

    @Positive
  void test(int[] arr, @LTLengthOf({"#1"}) int a) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int c = a + 1;
    @Positive
    @LTEqLengthOf({"arr"}) int c1 = a + 1;
    @Positive
    @LTLengthOf({"arr"}) int d = a;
    @Positive
    @LTLengthOf({"arr"}) int e = a + (-7);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int f = a + 7;
    @Positive
  }
    @Positive
}
