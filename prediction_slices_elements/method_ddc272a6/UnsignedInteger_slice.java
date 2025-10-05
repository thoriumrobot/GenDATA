// Source-based slice around line 210
// Method: <com.google.common.primitives.UnsignedInteger: BigInteger bigIntegerValue()>

   * Returns the value of this {@code UnsignedInteger} as a {@code double}, analogous to a widening
   * primitive conversion from {@code int} to {@code double}, and correctly rounded.
   */
  @Override
  public double doubleValue() {
    return longValue();
  }

  /** Returns the value of this {@code UnsignedInteger} as a {@link BigInteger}. */
  public BigInteger bigIntegerValue() {
    return BigInteger.valueOf(longValue());
  }

  /**
   * Compares this unsigned integer to another unsigned integer. Returns {@code 0} if they are
   * equal, a negative number if {@code this < other}, and a positive number if {@code this >
   * other}.
   */
  @Override
  public int compareTo(UnsignedInteger other) {
