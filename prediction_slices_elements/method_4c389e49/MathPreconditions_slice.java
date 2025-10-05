// Source-based slice around line 91
// Method: <com.google.common.math.MathPreconditions: void checkInRangeForRoundingInputs(boolean,double,RoundingMode)>

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
  }

  static void checkNoOverflow(boolean condition, String methodName, int a, int b) {
    if (!condition) {
      throw new ArithmeticException("overflow: " + methodName + "(" + a + ", " + b + ")");
    }
