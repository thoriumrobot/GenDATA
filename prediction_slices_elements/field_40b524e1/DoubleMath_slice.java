// Source-based slice around line 169
// Method: com.google.common.math.DoubleMath.MAX_LONG_AS_DOUBLE_PLUS_ONE

        MIN_LONG_AS_DOUBLE - z < 1.0 & z < MAX_LONG_AS_DOUBLE_PLUS_ONE, x, mode);
    return (long) z;
  }

  private static final double MIN_LONG_AS_DOUBLE = -0x1p63;
  /*
   * We cannot store Long.MAX_VALUE as a double without losing precision. Instead, we store
   * Long.MAX_VALUE + 1 == -Long.MIN_VALUE, and then offset all comparisons by 1.
   */
  private static final double MAX_LONG_AS_DOUBLE_PLUS_ONE = 0x1p63;

  /**
   * Returns the {@code BigInteger} value that is equal to {@code x} rounded with the specified
   * rounding mode, if possible.
   *
   * @throws ArithmeticException if
   *     <ul>
   *       <li>{@code x} is infinite or NaN
   *       <li>{@code x} is not a mathematical integer and {@code mode} is {@link
   *           RoundingMode#UNNECESSARY}
