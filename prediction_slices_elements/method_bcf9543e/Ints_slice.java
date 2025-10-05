// Source-based slice around line 95
// Method: <com.google.common.primitives.Ints: int checkedCast(long)>

   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated. Use {@link
   * Math#toIntExact(long)} instead, but be aware that that method throws {@link
   * ArithmeticException} rather than {@link IllegalArgumentException}.
   *
   * @param value any value in the range of the {@code int} type
   * @return the {@code int} value that equals {@code value}
   * @throws IllegalArgumentException if {@code value} is greater than {@link Integer#MAX_VALUE} or
   *     less than {@link Integer#MIN_VALUE}
   */
  public static int checkedCast(long value) {
    int result = (int) value;
    checkArgument(result == value, "Out of range: %s", value);
    return result;
  }

  /**
   * Returns the {@code int} nearest in value to {@code value}.
   *
   * @param value any {@code long} value
   * @return the same value cast to {@code int} if it is in the range of the {@code int} type,
