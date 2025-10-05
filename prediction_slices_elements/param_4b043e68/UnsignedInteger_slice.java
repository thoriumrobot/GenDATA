// Source-based slice around line 220
// Method: <com.google.common.primitives.UnsignedInteger: int compareTo(UnsignedInteger)>

    return BigInteger.valueOf(longValue());
  }

  /**
   * Compares this unsigned integer to another unsigned integer. Returns {@code 0} if they are
   * equal, a negative number if {@code this < other}, and a positive number if {@code this >
   * other}.
   */
  @Override
  public int compareTo(UnsignedInteger other) {
    checkNotNull(other);
    return compare(value, other.value);
  }

  @Override
  public int hashCode() {
    return value;
  }

  @Override
