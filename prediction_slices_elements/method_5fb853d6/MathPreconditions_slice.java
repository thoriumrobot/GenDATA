// Source-based slice around line 85
// Method: <com.google.common.math.MathPreconditions: void checkRoundingUnnecessary(boolean)>


  @CanIgnoreReturnValue
  static double checkNonNegative(String role, double x) {
    if (!(x >= 0)) { // not x < 0, to work with NaN.
      throw new IllegalArgumentException(role + " (" + x + ") must be >= 0");
    }
    return x;
  }

  static void checkRoundingUnnecessary(boolean condition) {
    if (!condition) {
      throw new ArithmeticException("mode was UNNECESSARY, but rounding was necessary");
    }
  }

  static void checkInRangeForRoundingInputs(boolean condition, double input, RoundingMode mode) {
    if (!condition) {
      throw new ArithmeticException(
          "rounded value is out of range for input " + input + " and rounding mode " + mode);
    }
