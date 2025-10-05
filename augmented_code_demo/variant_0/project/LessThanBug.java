    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.common.value.qual.IntVal;

    @Positive
public class LessThanBug {

    @Positive
  void call() {
    @Positive
    bug(30, 1);
    @Positive
  }

    @Positive
  void bug(@IntRange(to = 42) int a, @IntVal(1) int c) {
    // :: error: (assignment)
    @Positive
    @LessThan("c") int x = a;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
