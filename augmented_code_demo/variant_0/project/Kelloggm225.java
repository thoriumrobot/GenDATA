    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class Kelloggm225 {
    @Positive
  void method(int @MinLen(1) [] bar) {
    @Positive
    foo(bar, 0, bar.length);
    @Positive
  }

    @Positive
  void foo(
    @Positive
      int @MinLen(1) [] bar,
    @Positive
      @IndexFor("#1") @LessThan("#3") int start,
    @Positive
      @IndexOrHigh("#1") int end) {}
    @Positive
}

// CFWR semantic augmentation - variant 0
