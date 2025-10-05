    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class BottomValTest {
    @Positive
  @NonNegative int foo(@BottomVal int bottom) {
    @Positive
    return bottom;
    @Positive
  }

    @Positive
  @Positive int bar(@BottomVal int bottom) {
    @Positive
    return bottom;
    @Positive
  }

    @Positive
  @LTLengthOf("#1") int baz(int[] a, @BottomVal int bottom) {
    @Positive
    return bottom;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
