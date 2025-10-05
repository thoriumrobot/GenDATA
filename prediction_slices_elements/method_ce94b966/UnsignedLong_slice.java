// Source-based slice around line 210
// Method: <com.google.common.primitives.UnsignedLong: double doubleValue()>

    // So we can ignore the bottom 8, except for rounding. See doubleValue() for more.
    return (float) ((value >>> 1) | (value & 1)) * 2f;
  }

  /**
   * Returns the value of this {@code UnsignedLong} as a {@code double}, analogous to a widening
   * primitive conversion from {@code long} to {@code double}, and correctly rounded.
   */
  @Override
  public double doubleValue() {
    if (value >= 0) {
      return (double) value;
    }
    // The top bit is set, which means that the double value is going to come from the top 53 bits.
    // So we can ignore the bottom 11, except for rounding. We can unsigned-shift right 1, aka
    // unsigned-divide by 2, and convert that. Then we'll get exactly half of the desired double
    // value. But in the specific case where the bottom two bits of the original number are 01, we
    // want to replace that with 1 in the shifted value for correct rounding.
    return (double) ((value >>> 1) | (value & 1)) * 2.0;
  }
