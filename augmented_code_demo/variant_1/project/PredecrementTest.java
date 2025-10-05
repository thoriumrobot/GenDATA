    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class PredecrementTest {

    @Positive
  public static void warningForLoop(int @MinLen(1) [] a) {
    @Positive
    for (int i = a.length; --i >= 0; ) {
    @Positive
      a[i] = 0;
    @Positive
    }
    @Positive
  }

    @Positive
  public static void warningIfStatement(int @MinLen(1) [] a) {
    @Positive
    int i = a.length;
    @Positive
    if (--i >= 0) {
    @Positive
      a[i] = 0;
    @Positive
    }
    @Positive
  }

    @Positive
  public static void warningIfStatementRange1(
    @Positive
      int @MinLen(1) [] a, @IntRange(from = 1, to = 1) int j) {
    @Positive
    int i = j;
    @Positive
    if (--i >= 0) {
    @Positive
      a[i] = 0;
    @Positive
    }
    @Positive
  }

    @Positive
  public static void warningIfStatementVal1(int @MinLen(1) [] a, @IntVal(1) int j) {
    @Positive
    int i = j;
    @Positive
    if (--i >= 0) {
    @Positive
      a[i] = 0;
    @Positive
    }
    @Positive
  }

    @Positive
  public static void warningIfStatementRange12(
    @Positive
      int @MinLen(2) [] a, @IntRange(from = 1, to = 2) int j) {
    @Positive
    int i = j;
    @Positive
    if (--i >= 0) {
    @Positive
      a[i] = 0;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
