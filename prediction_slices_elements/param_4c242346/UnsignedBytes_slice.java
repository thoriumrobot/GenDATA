// Source-based slice around line 98
// Method: <com.google.common.primitives.UnsignedBytes: byte checkedCast(long)>

  /**
   * Returns the {@code byte} value that, when treated as unsigned, is equal to {@code value}, if
   * possible.
   *
   * @param value a value between 0 and 255 inclusive
   * @return the {@code byte} value that, when treated as unsigned, equals {@code value}
   * @throws IllegalArgumentException if {@code value} is negative or greater than 255
   */
  @CanIgnoreReturnValue
  public static byte checkedCast(long value) {
    checkArgument(value >> Byte.SIZE == 0, "out of range: %s", value);
    return (byte) value;
  }

  /**
   * Returns the {@code byte} value that, when treated as unsigned, is nearest in value to {@code
   * value}.
   *
   * @param value any {@code long} value
   * @return {@code (byte) 255} if {@code value >= 255}, {@code (byte) 0} if {@code value <= 0}, and
