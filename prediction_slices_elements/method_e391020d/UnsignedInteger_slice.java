// Source-based slice around line 169
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger mod(UnsignedInteger)>

    return fromIntBits(UnsignedInts.divide(value, checkNotNull(val).value));
  }

  /**
   * Returns this mod {@code val}.
   *
   * @throws ArithmeticException if {@code val} is zero
   * @since 14.0
   */
  public UnsignedInteger mod(UnsignedInteger val) {
    return fromIntBits(UnsignedInts.remainder(value, checkNotNull(val).value));
  }

  /**
   * Returns the value of this {@code UnsignedInteger} as an {@code int}. This is an inverse
   * operation to {@link #fromIntBits}.
   *
   * <p>Note that if this {@code UnsignedInteger} holds a value {@code >= 2^31}, the returned value
   * will be equal to {@code this - 2^32}.
   */
