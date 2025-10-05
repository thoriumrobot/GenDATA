// Source-based slice around line 402
// Method: <com.google.common.math.IntMath: int gcd(int,int)>

    return Math.floorMod(x, m);
  }

  /**
   * Returns the greatest common divisor of {@code a, b}. Returns {@code 0} if {@code a == 0 && b ==
   * 0}.
   *
   * @throws IllegalArgumentException if {@code a < 0} or {@code b < 0}
   */
  public static int gcd(int a, int b) {
    /*
     * The reason we require both arguments to be >= 0 is because otherwise, what do you return on
     * gcd(0, Integer.MIN_VALUE)? BigInteger.gcd would return positive 2^31, but positive 2^31 isn't
     * an int.
     */
    checkNonNegative("a", a);
    checkNonNegative("b", b);
    if (a == 0) {
      // 0 % b == 0, so b divides a, but the converse doesn't hold.
      // BigInteger.gcd is consistent with this decision.
