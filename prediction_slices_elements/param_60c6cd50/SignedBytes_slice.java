// Source-based slice around line 57
// Method: <com.google.common.primitives.SignedBytes: byte checkedCast(long)>


  /**
   * Returns the {@code byte} value that is equal to {@code value}, if possible.
   *
   * @param value any value in the range of the {@code byte} type
   * @return the {@code byte} value that equals {@code value}
   * @throws IllegalArgumentException if {@code value} is greater than {@link Byte#MAX_VALUE} or
   *     less than {@link Byte#MIN_VALUE}
   */
  public static byte checkedCast(long value) {
    byte result = (byte) value;
    checkArgument(result == value, "Out of range: %s", value);
    return result;
  }

  /**
   * Returns the {@code byte} nearest in value to {@code value}.
   *
   * @param value any {@code long} value
   * @return the same value cast to {@code byte} if it is in the range of the {@code byte} type,
