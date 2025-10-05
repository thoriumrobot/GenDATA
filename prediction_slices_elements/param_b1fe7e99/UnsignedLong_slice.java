// Source-based slice around line 160
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong dividedBy(UnsignedLong)>

  public UnsignedLong times(UnsignedLong val) {
    return fromLongBits(value * checkNotNull(val).value);
  }

  /**
   * Returns the result of dividing this by {@code val}.
   *
   * @since 14.0
   */
  public UnsignedLong dividedBy(UnsignedLong val) {
    return fromLongBits(UnsignedLongs.divide(value, checkNotNull(val).value));
  }

  /**
   * Returns this modulo {@code val}.
   *
   * @since 14.0
   */
  public UnsignedLong mod(UnsignedLong val) {
    return fromLongBits(UnsignedLongs.remainder(value, checkNotNull(val).value));
