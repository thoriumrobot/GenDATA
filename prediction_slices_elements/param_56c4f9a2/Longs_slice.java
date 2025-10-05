// Source-based slice around line 240
// Method: <com.google.common.primitives.Longs: long constrainToRange(long,long,long)>

   * <p><b>Java 21+ users:</b> Use {@code Math.clamp} instead. Note that that method is capable of
   * constraining a {@code long} input to an {@code int} range.
   *
   * @param value the {@code long} value to constrain
   * @param min the lower bound (inclusive) of the range to constrain {@code value} to
   * @param max the upper bound (inclusive) of the range to constrain {@code value} to
   * @throws IllegalArgumentException if {@code min > max}
   * @since 21.0
   */
  public static long constrainToRange(long value, long min, long max) {
    checkArgument(min <= max, "min (%s) must be less than or equal to max (%s)", min, max);
    return Math.min(Math.max(value, min), max);
  }

  /**
   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new long[] {a, b}, new long[] {}, new long[] {c}} returns the array {@code {a, b, c}}.
   *
   * @param arrays zero or more {@code long} arrays
   * @return a single array containing all the values from the source arrays, in order
