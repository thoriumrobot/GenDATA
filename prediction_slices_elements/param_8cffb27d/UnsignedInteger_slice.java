// Source-based slice around line 159
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger dividedBy(UnsignedInteger)>

    return fromIntBits(value * checkNotNull(val).value);
  }

  /**
   * Returns the result of dividing this by {@code val}.
   *
   * @throws ArithmeticException if {@code val} is zero
   * @since 14.0
   */
  public UnsignedInteger dividedBy(UnsignedInteger val) {
    return fromIntBits(UnsignedInts.divide(value, checkNotNull(val).value));
  }

  /**
   * Returns this mod {@code val}.
   *
   * @throws ArithmeticException if {@code val} is zero
   * @since 14.0
   */
  public UnsignedInteger mod(UnsignedInteger val) {
