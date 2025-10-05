// Source-based slice around line 151
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong times(UnsignedLong)>

    return fromLongBits(this.value - checkNotNull(val).value);
  }

  /**
   * Returns the result of multiplying this and {@code val}. If the result would have more than 64
   * bits, returns the low 64 bits of the result.
   *
   * @since 14.0
   */
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
