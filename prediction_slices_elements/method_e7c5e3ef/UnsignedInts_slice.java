// Source-based slice around line 93
// Method: <com.google.common.primitives.UnsignedInts: int checkedCast(long)>

   * Returns the {@code int} value that, when treated as unsigned, is equal to {@code value}, if
   * possible.
   *
   * @param value a value between 0 and 2<sup>32</sup>-1 inclusive
   * @return the {@code int} value that, when treated as unsigned, equals {@code value}
   * @throws IllegalArgumentException if {@code value} is negative or greater than or equal to
   *     2<sup>32</sup>
   * @since 21.0
   */
  public static int checkedCast(long value) {
    checkArgument((value >> Integer.SIZE) == 0, "out of range: %s", value);
    return (int) value;
  }

  /**
   * Returns the {@code int} value that, when treated as unsigned, is nearest in value to {@code
   * value}.
   *
   * @param value any {@code long} value
   * @return {@code 2^32 - 1} if {@code value >= 2^32}, {@code 0} if {@code value <= 0}, and {@code
