/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class DefaultingForEach {
// Test case for issue #4248: https://github.com/typetools/checker-framework/issues/4248.
// This test exposed a crash in the original version of the fix.

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.framework.qual.DefaultQualifier;

    @Positive
class DefaultForEach {

    @Positive
  @DefaultQualifier(NonNegative.class)
    @Positive
  static int[] foo() {
    @Positive
    throw new RuntimeException();
    @Positive
  }

    @Positive
  void bar() {
    @Positive
    for (Integer p : foo()) {
      // :: error: (assignment)
    @Positive
      @Positive int x = p;
    @Positive
      @NonNegative int y = p;
    @Positive
    }
    @Positive
  }
    @Positive
}

}