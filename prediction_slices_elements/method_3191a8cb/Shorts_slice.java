// Source-based slice around line 90
// Method: <com.google.common.primitives.Shorts: short checkedCast(long)>


  /**
   * Returns the {@code short} value that is equal to {@code value}, if possible.
   *
   * @param value any value in the range of the {@code short} type
   * @return the {@code short} value that equals {@code value}
   * @throws IllegalArgumentException if {@code value} is greater than {@link Short#MAX_VALUE} or
   *     less than {@link Short#MIN_VALUE}
   */
  public static short checkedCast(long value) {
    short result = (short) value;
    checkArgument(result == value, "Out of range: %s", value);
    return result;
  }

  /**
   * Returns the {@code short} nearest in value to {@code value}.
   *
   * @param value any {@code long} value
   * @return the same value cast to {@code short} if it is in the range of the {@code short} type,
