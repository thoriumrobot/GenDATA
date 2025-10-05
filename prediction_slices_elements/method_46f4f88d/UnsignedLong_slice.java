// Source-based slice around line 175
// Method: <com.google.common.primitives.UnsignedLong: int intValue()>

   *
   * @since 14.0
   */
  public UnsignedLong mod(UnsignedLong val) {
    return fromLongBits(UnsignedLongs.remainder(value, checkNotNull(val).value));
  }

  /** Returns the value of this {@code UnsignedLong} as an {@code int}. */
  @Override
  public int intValue() {
    return (int) value;
  }

  /**
   * Returns the value of this {@code UnsignedLong} as a {@code long}. This is an inverse operation
   * to {@link #fromLongBits}.
   *
   * <p>Note that if this {@code UnsignedLong} holds a value {@code >= 2^63}, the returned value
   * will be equal to {@code this - 2^64}.
   */
