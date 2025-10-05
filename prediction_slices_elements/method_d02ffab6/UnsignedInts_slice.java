// Source-based slice around line 107
// Method: <com.google.common.primitives.UnsignedInts: int saturatedCast(long)>

  /**
   * Returns the {@code int} value that, when treated as unsigned, is nearest in value to {@code
   * value}.
   *
   * @param value any {@code long} value
   * @return {@code 2^32 - 1} if {@code value >= 2^32}, {@code 0} if {@code value <= 0}, and {@code
   *     value} cast to {@code int} otherwise
   * @since 21.0
   */
  public static int saturatedCast(long value) {
    if (value <= 0) {
      return 0;
    } else if (value >= (1L << 32)) {
      return -1;
    } else {
      return (int) value;
    }
  }

  /**
