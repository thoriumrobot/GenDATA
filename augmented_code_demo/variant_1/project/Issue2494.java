// Test case for issue #2494: http://tinyurl.com/cfissue/2494

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public final class Issue2494 {

    @Positive
  static final long @MinLen(1) [] factorials = {
    @Positive
    1L,
    @Positive
    1L,
    @Positive
    1L * 2,
    @Positive
    1L * 2 * 3,
    @Positive
    1L * 2 * 3 * 4,
    @Positive
    1L * 2 * 3 * 4 * 5,
    @Positive
    1L * 2 * 3 * 4 * 5 * 6,
    @Positive
    1L * 2 * 3 * 4 * 5 * 6 * 7
    @Positive
  };

    @Positive
  static void binomialA(
    @Positive
      @NonNegative @LTLengthOf("Issue2494.factorials") int n,
    @Positive
      @NonNegative @LessThan("#1 + 1") int k) {
    @Positive
    @IndexFor("factorials") int j = k;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
