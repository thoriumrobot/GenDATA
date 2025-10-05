// Source-based slice around line 187
// Method: <com.google.common.primitives.UnsignedInteger: long longValue()>

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
   * Returns the value of this {@code UnsignedInteger} as a {@code float}, analogous to a widening
   * primitive conversion from {@code int} to {@code float}, and correctly rounded.
   */
  @Override
  public float floatValue() {
    return longValue();
