// Source-based slice around line 131
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong plus(UnsignedLong)>

    return fromLongBits(UnsignedLongs.parseUnsignedLong(string, radix));
  }

  /**
   * Returns the result of adding this and {@code val}. If the result would have more than 64 bits,
   * returns the low 64 bits of the result.
   *
   * @since 14.0
   */
  public UnsignedLong plus(UnsignedLong val) {
    return fromLongBits(this.value + checkNotNull(val).value);
  }

  /**
   * Returns the result of subtracting this and {@code val}. If the result would have more than 64
   * bits, returns the low 64 bits of the result.
   *
   * @since 14.0
   */
  public UnsignedLong minus(UnsignedLong val) {
