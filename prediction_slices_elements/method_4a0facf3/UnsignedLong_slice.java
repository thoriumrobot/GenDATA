// Source-based slice around line 196
// Method: <com.google.common.primitives.UnsignedLong: float floatValue()>

  public long longValue() {
    return value;
  }

  /**
   * Returns the value of this {@code UnsignedLong} as a {@code float}, analogous to a widening
   * primitive conversion from {@code long} to {@code float}, and correctly rounded.
   */
  @Override
  public float floatValue() {
    if (value >= 0) {
      return (float) value;
    }
    // The top bit is set, which means that the float value is going to come from the top 24 bits.
    // So we can ignore the bottom 8, except for rounding. See doubleValue() for more.
    return (float) ((value >>> 1) | (value & 1)) * 2f;
  }

  /**
   * Returns the value of this {@code UnsignedLong} as a {@code double}, analogous to a widening
