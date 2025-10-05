// Source-based slice around line 223
// Method: <com.google.common.primitives.UnsignedLong: BigInteger bigIntegerValue()>

    // The top bit is set, which means that the double value is going to come from the top 53 bits.
    // So we can ignore the bottom 11, except for rounding. We can unsigned-shift right 1, aka
    // unsigned-divide by 2, and convert that. Then we'll get exactly half of the desired double
    // value. But in the specific case where the bottom two bits of the original number are 01, we
    // want to replace that with 1 in the shifted value for correct rounding.
    return (double) ((value >>> 1) | (value & 1)) * 2.0;
  }

  /** Returns the value of this {@code UnsignedLong} as a {@link BigInteger}. */
  public BigInteger bigIntegerValue() {
    BigInteger bigInt = BigInteger.valueOf(value & UNSIGNED_MASK);
    if (value < 0) {
      bigInt = bigInt.setBit(Long.SIZE - 1);
    }
    return bigInt;
  }

  @Override
  public int compareTo(UnsignedLong o) {
    checkNotNull(o);
