// Test case for issue #2541: https://github.com/typetools/checker-framework/issues/2541

    @Positive
public class LessThanConstantAddition {

    @Positive
  public static void checkedPow(int b) {
    @Positive
    if (b <= 2) {
    @Positive
      int c = (int) b;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
