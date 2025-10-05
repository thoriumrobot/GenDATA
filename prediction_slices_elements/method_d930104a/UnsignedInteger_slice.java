// Source-based slice around line 67
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger fromIntBits(int)>

   * interpreted as a normal bit, and all other bits are treated as usual.
   *
   * <p>If the argument is nonnegative, the returned result will be equal to {@code bits},
   * otherwise, the result will be equal to {@code 2^32 + bits}.
   *
   * <p>To represent unsigned decimal constants, consider {@link #valueOf(long)} instead.
   *
   * @since 14.0
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
