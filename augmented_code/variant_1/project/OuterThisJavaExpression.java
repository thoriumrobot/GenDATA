/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue #4558: https://tinyurl.com/cfissue/4558

// @skip-test until the issue is fixed

    @Positive
import org.checkerframework.checker.index.qual.SameLen;

    @Positive
public abstract class OuterThisJavaExpression {

    @Positive
  String s;

    @Positive
  OuterThisJavaExpression(String s) {
    @Positive
    this.s = s;
    @Positive
  }

    @Positive
  final class Inner {

    @Positive
    String s = "different from " + OuterThisJavaExpression.this.s;

    @Positive
    @SameLen("s") String f1() {
    @Positive
      return s;
    @Positive
    }

    @Positive
    @SameLen("s") String f2() {
    @Positive
      return this.s;
    @Positive
    }

    @Positive
    @SameLen("s") String f3() {
      // :: error: (return)
    @Positive
      return OuterThisJavaExpression.this.s;
    @Positive
    }

    @Positive
    @SameLen("this.s") String f4() {
    @Positive
      return s;
    @Positive
    }

    @Positive
    @SameLen("this.s") String f5() {
    @Positive
      return this.s;
    @Positive
    }

    @Positive
    @SameLen("this.s") String f6() {
      // :: error: (return)
    @Positive
      return OuterThisJavaExpression.this.s;
    @Positive
    }

    @Positive
    @SameLen("OuterThisJavaExpression.this.s") String f7() {
      // :: error: (return)
    @Positive
      return s;
    @Positive
    }

    @Positive
    @SameLen("OuterThisJavaExpression.this.s") String f8() {
      // :: error: (return)
    @Positive
      return this.s;
    @Positive
    }

    @Positive
    @SameLen("OuterThisJavaExpression.this.s") String f9() {
    @Positive
      return OuterThisJavaExpression.this.s;
    @Positive
    }
    @Positive
  }
    @Positive
}
