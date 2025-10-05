// Source-based slice around line 75
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger valueOf(long)>

   */
  public static UnsignedInteger fromIntBits(int bits) {
    return new UnsignedInteger(bits);
  }

  /**
   * Returns an {@code UnsignedInteger} that is equal to {@code value}, if possible. The inverse
   * operation of {@link #longValue()}.
   */
  public static UnsignedInteger valueOf(long value) {
    checkArgument(
        (value & INT_MASK) == value,
        "value (%s) is outside the range for an unsigned integer value",
        value);
    return fromIntBits((int) value);
  }

  /**
   * Returns a {@code UnsignedInteger} representing the same value as the specified {@link
   * BigInteger}. This is the inverse operation of {@link #bigIntegerValue()}.
