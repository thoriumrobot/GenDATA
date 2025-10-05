// Source-based slice around line 540
// Method: <com.google.common.math.IntMath: int saturatedAdd(int,int)>

    }
  }

  /**
   * Returns the sum of {@code a} and {@code b} unless it would overflow or underflow in which case
   * {@code Integer.MAX_VALUE} or {@code Integer.MIN_VALUE} is returned, respectively.
   *
   * @since 20.0
   */
  public static int saturatedAdd(int a, int b) {
    return Ints.saturatedCast((long) a + b);
  }

  /**
   * Returns the difference of {@code a} and {@code b} unless it would overflow or underflow in
   * which case {@code Integer.MAX_VALUE} or {@code Integer.MIN_VALUE} is returned, respectively.
   *
   * @since 20.0
   */
  public static int saturatedSubtract(int a, int b) {
