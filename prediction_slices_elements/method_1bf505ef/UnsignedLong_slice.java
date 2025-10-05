// Source-based slice around line 187
// Method: <com.google.common.primitives.UnsignedLong: long longValue()>


  /**
   * Returns the value of this {@code UnsignedLong} as a {@code long}. This is an inverse operation
   * to {@link #fromLongBits}.
   *
   * <p>Note that if this {@code UnsignedLong} holds a value {@code >= 2^63}, the returned value
   * will be equal to {@code this - 2^64}.
   */
  @Override
  public long longValue() {
    return value;
  }

  /**
   * Returns the value of this {@code UnsignedLong} as a {@code float}, analogous to a widening
   * primitive conversion from {@code long} to {@code float}, and correctly rounded.
   */
  @Override
  public float floatValue() {
    if (value >= 0) {
