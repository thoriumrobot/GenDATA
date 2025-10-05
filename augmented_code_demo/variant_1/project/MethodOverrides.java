// This class should not issues any errors from the value checker.
// The index checker should issue the errors instead.

// There is a copy of this test at checker/tests/value-index-interaction/MethodOverrides.java,
// which does not include expected failures.

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;

    @Positive
public class MethodOverrides {
    @Positive
  @GTENegativeOne int read() {
    @Positive
    return -1;
    @Positive
  }
    @Positive
}

    @Positive
class MethodOverrides2 extends MethodOverrides {
  // :: error: (override.return)
    @Positive
  int read() {
    @Positive
    return -1;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
