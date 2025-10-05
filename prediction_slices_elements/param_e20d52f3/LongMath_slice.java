// Source-based slice around line 488
// Method: <com.google.common.math.LongMath: long gcd(long,long)>

    return Math.floorMod(x, m);
  }

  /**
   * Returns the greatest common divisor of {@code a, b}. Returns {@code 0} if {@code a == 0 && b ==
   * 0}.
   *
   * @throws IllegalArgumentException if {@code a < 0} or {@code b < 0}
   */
  public static long gcd(long a, long b) {
    /*
     * The reason we require both arguments to be >= 0 is because otherwise, what do you return on
     * gcd(0, Long.MIN_VALUE)? BigInteger.gcd would return positive 2^63, but positive 2^63 isn't an
     * int.
     */
    checkNonNegative("a", a);
    checkNonNegative("b", b);
    if (a == 0) {
      // 0 % b == 0, so b divides a, but the converse doesn't hold.
      // BigInteger.gcd is consistent with this decision.
