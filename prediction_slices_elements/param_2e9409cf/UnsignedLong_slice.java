// Source-based slice around line 169
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong mod(UnsignedLong)>

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
  }

  /** Returns the value of this {@code UnsignedLong} as an {@code int}. */
  @Override
  public int intValue() {
    return (int) value;
  }

  /**
