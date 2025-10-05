// Source-based slice around line 250
// Method: <com.google.common.primitives.UnsignedInteger: String toString(int)>

  public String toString() {
    return toString(10);
  }

  /**
   * Returns a string representation of the {@code UnsignedInteger} value, in base {@code radix}. If
   * {@code radix < Character.MIN_RADIX} or {@code radix > Character.MAX_RADIX}, the radix {@code
   * 10} is used.
   */
  public String toString(int radix) {
    return UnsignedInts.toString(value, radix);
  }
}
