// Source-based slice around line 262
// Method: <com.google.common.primitives.UnsignedLong: String toString(int)>

  public String toString() {
    return UnsignedLongs.toString(value);
  }

  /**
   * Returns a string representation of the {@code UnsignedLong} value, in base {@code radix}. If
   * {@code radix < Character.MIN_RADIX} or {@code radix > Character.MAX_RADIX}, the radix {@code
   * 10} is used.
   */
  public String toString(int radix) {
    return UnsignedLongs.toString(value, radix);
  }
}
