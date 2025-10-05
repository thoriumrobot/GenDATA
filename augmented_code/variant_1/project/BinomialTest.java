/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class BinomialTest {

    @Positive
  static final long @MinLen(1) [] factorials = {1L, 1L, 2 * 1L};

    @Positive
  public static long binomial(
    @Positive
      @NonNegative @LTLengthOf("BinomialTest.factorials") int n,
    @Positive
      @NonNegative @LessThan("#1 + 1") int k) {
    @Positive
    return factorials[k];
    @Positive
  }

    @Positive
  public static void binomial0(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @Positive
    @LTLengthOf(value = "factorials", offset = "1") int i = k;
    @Positive
  }

    @Positive
  public static void binomial0Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "2") int i = k;
    @Positive
  }

    @Positive
  public static void binomial0Weak(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1") int k) {
    @Positive
    @LTLengthOf("factorials") int i = k;
    @Positive
  }

    @Positive
  public static void binomial1(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 1") int k) {
    @Positive
    @LTLengthOf("factorials") int i = k;
    @Positive
  }

    @Positive
  public static void binomial1Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 1") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "1") int i = k;
    @Positive
  }

    @Positive
  public static void binomial2(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 2") int k) {
    @Positive
    @LTLengthOf(value = "factorials", offset = "-1") int i = k;
    @Positive
  }

    @Positive
  public static void binomial2Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 + 2") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "0") int i = k;
    @Positive
  }

    @Positive
  public static void binomial_1(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 1") int k) {
    @Positive
    @LTLengthOf(value = "factorials", offset = "2") int i = k;
    @Positive
  }

    @Positive
  public static void binomial_1Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 1") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "3") int i = k;
    @Positive
  }

    @Positive
  public static void binomial_2(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 2") int k) {
    @Positive
    @LTLengthOf(value = "factorials", offset = "3") int i = k;
    @Positive
  }

    @Positive
  public static void binomial_2Error(
    @Positive
      @LTLengthOf("BinomialTest.factorials") int n, @LessThan("#1 - 2") int k) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "factorials", offset = "4") int i = k;
    @Positive
  }
    @Positive
}
