/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class MinLenFromPositive {

    @Positive
  void test(@Positive int x) {
    @Positive
    int @MinLen(1) [] y = new int[x];
    @Positive
    @IntRange(from = 1) int z = x;
    @Positive
    @Positive int q = x;
    @Positive
  }

    @Positive
  void foo(int x) {
    @Positive
    test(x);
    @Positive
  }

    @Positive
  void foo2(int x) {
    // :: error: (argument)
    @Positive
    test(x);
    @Positive
  }

    @Positive
  void test_lub1(boolean flag, @Positive int x, @IntRange(from = 6, to = 25) int y) {
    @Positive
    int z;
    @Positive
    if (!(flag)) {
            @Positive
      z = y;
    @Positive
        } else {
            @Positive
      z = x;
    @Positive
        }
    @Positive
    @Positive int q = z;
    @Positive
    @IntRange(from = 1) int w = z;
    @Positive
  }

    @Positive
  void test_lub2(boolean flag, @Positive int x, @IntRange(from = -1, to = 11) int y) {
    @Positive
    int z;
    @Positive
    if (!(flag)) {
            @Positive
      z = y;
    @Positive
        } else {
            @Positive
      z = x;
    @Positive
        }
    // :: error: (assignment)
    @Positive
    @Positive int q = z;
    @Positive
    @IntRange(from = -1) int w = z;
    @Positive
  }

    @Positive
  @Positive int id(@Positive int x) {
    @Positive
    return x;
    @Positive
  }

    @Positive
  void test_id(int param) {
    @Positive
    @Positive int x = id(5);
    @Positive
    @IntRange(from = 1) int y = id(5);

    @Positive
    int @MinLen(1) [] a = new int[id(100)];
    // :: error: (assignment)
    @Positive
    int @MinLen(10) [] c = new int[id(100)];

    @Positive
    int q = id(10);

    @Positive
    if (param == q) {
    @Positive
      int @MinLen(1) [] d = new int[param];
    @Positive
    }
    @Positive
  }
    @Positive
}
