// Source-based slice around line 560
// Method: <com.google.common.math.IntMath: int saturatedMultiply(int,int)>

    return Ints.saturatedCast((long) a - b);
  }

  /**
   * Returns the product of {@code a} and {@code b} unless it would overflow or underflow in which
   * case {@code Integer.MAX_VALUE} or {@code Integer.MIN_VALUE} is returned, respectively.
   *
   * @since 20.0
   */
  public static int saturatedMultiply(int a, int b) {
    return Ints.saturatedCast((long) a * b);
  }

  /**
   * Returns the {@code b} to the {@code k}th power, unless it would overflow or underflow in which
   * case {@code Integer.MAX_VALUE} or {@code Integer.MIN_VALUE} is returned, respectively.
   *
   * @since 20.0
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
