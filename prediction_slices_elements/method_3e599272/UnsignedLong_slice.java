// Source-based slice around line 67
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong fromLongBits(long)>

   *
   * <p>If the argument is nonnegative, the returned result will be equal to {@code bits},
   * otherwise, the result will be equal to {@code 2^64 + bits}.
   *
   * <p>To represent decimal constants less than {@code 2^63}, consider {@link #valueOf(long)}
   * instead.
   *
   * @since 14.0
   */
  public static UnsignedLong fromLongBits(long bits) {
    // TODO(lowasser): consider caching small values, like Long.valueOf
    return new UnsignedLong(bits);
  }

  /**
   * Returns an {@code UnsignedLong} representing the same value as the specified {@code long}.
   *
   * @throws IllegalArgumentException if {@code value} is negative
   * @since 14.0
   */
