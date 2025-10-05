// Source-based slice around line 181
// Method: <com.google.common.primitives.UnsignedInteger: int intValue()>


  /**
   * Returns the value of this {@code UnsignedInteger} as an {@code int}. This is an inverse
   * operation to {@link #fromIntBits}.
   *
   * <p>Note that if this {@code UnsignedInteger} holds a value {@code >= 2^31}, the returned value
   * will be equal to {@code this - 2^32}.
   */
  @Override
  public int intValue() {
    return value;
  }

  /** Returns the value of this {@code UnsignedInteger} as a {@code long}. */
  @Override
  public long longValue() {
    return toLong(value);
  }

  /**
