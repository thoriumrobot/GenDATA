// Source-based slice around line 79
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong valueOf(long)>

  }

  /**
   * Returns an {@code UnsignedLong} representing the same value as the specified {@code long}.
   *
   * @throws IllegalArgumentException if {@code value} is negative
   * @since 14.0
   */
  @CanIgnoreReturnValue
  public static UnsignedLong valueOf(long value) {
    checkArgument(value >= 0, "value (%s) is outside the range for an unsigned long value", value);
    return fromLongBits(value);
  }

  /**
   * Returns a {@code UnsignedLong} representing the same value as the specified {@code BigInteger}.
   * This is the inverse operation of {@link #bigIntegerValue()}.
   *
   * @throws IllegalArgumentException if {@code value} is negative or {@code value >= 2^64}
   */
