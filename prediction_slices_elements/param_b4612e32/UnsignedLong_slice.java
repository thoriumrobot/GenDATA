// Source-based slice around line 141
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong minus(UnsignedLong)>

    return fromLongBits(this.value + checkNotNull(val).value);
  }

  /**
   * Returns the result of subtracting this and {@code val}. If the result would have more than 64
   * bits, returns the low 64 bits of the result.
   *
   * @since 14.0
   */
  public UnsignedLong minus(UnsignedLong val) {
    return fromLongBits(this.value - checkNotNull(val).value);
  }

  /**
   * Returns the result of multiplying this and {@code val}. If the result would have more than 64
   * bits, returns the low 64 bits of the result.
   *
   * @since 14.0
   */
  public UnsignedLong times(UnsignedLong val) {
