// Source-based slice around line 253
// Method: <com.google.common.primitives.UnsignedLong: String toString()>

    if (obj instanceof UnsignedLong) {
      UnsignedLong other = (UnsignedLong) obj;
      return value == other.value;
    }
    return false;
  }

  /** Returns a string representation of the {@code UnsignedLong} value, in base 10. */
  @Override
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
