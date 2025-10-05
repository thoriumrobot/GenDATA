/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for kelloggm 218
// https://github.com/kelloggm/checker-framework/issues/218

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class Modulo {
    @Positive
  void m1(Object[] a, @IndexOrHigh("#1") int i, @NonNegative int j) {
    @Positive
    @IndexFor("a") int k = j % i;
    @Positive
  }

    @Positive
  void m1p(Object[] a, @Positive @LTEqLengthOf("#1") int i, @Positive int j) {
    @Positive
    @IndexFor("a") int k = j % i;
    @Positive
  }

    @Positive
  void m2(Object[] a, int i, @IndexFor("#1") int j) {
    @Positive
    @IndexFor("a") int k = j % i;
    @Positive
  }

    @Positive
  void m2(Object[] a, Object[] b, @IndexFor("#1") int i, @IndexFor("#2") int j) {
    @Positive
    @IndexFor({"a", "b"}) int k = j % i;
    @Positive
  }
    @Positive
}
