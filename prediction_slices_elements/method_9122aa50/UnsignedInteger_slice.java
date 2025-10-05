// Source-based slice around line 241
// Method: <com.google.common.primitives.UnsignedInteger: String toString()>

    if (obj instanceof UnsignedInteger) {
      UnsignedInteger other = (UnsignedInteger) obj;
      return value == other.value;
    }
    return false;
  }

  /** Returns a string representation of the {@code UnsignedInteger} value, in base 10. */
  @Override
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
