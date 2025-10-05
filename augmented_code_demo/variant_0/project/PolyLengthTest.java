    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class PolyLengthTest {
    @Positive
  int @PolyLength [] id(int @PolyLength [] a) {
    @Positive
    return a;
    @Positive
  }

    @Positive
  int @SameLen("#2") [] test0(int @SameLen("#2") [] a, int @SameLen("#1") [] b) {
    @Positive
    return id(a);
    @Positive
  }

    @Positive
  int @ArrayLen(3) [] test1(int @ArrayLen(3) [] a) {
    @Positive
    return id(a);
    @Positive
  }

    @Positive
  int @MinLen(3) [] test2(int @MinLen(3) [] a) {
    @Positive
    return id(a);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
