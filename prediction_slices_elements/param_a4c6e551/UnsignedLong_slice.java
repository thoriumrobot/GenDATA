// Source-based slice around line 232
// Method: <com.google.common.primitives.UnsignedLong: int compareTo(UnsignedLong)>

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
    return UnsignedLongs.compare(value, o.value);
  }

  @Override
  public int hashCode() {
    return Long.hashCode(value);
  }

  @Override
