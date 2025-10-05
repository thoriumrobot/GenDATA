// Source-based slice around line 381
// Method: <com.google.common.math.BigIntegerMath: BigInteger factorial(int)>

   *
   * <p><b>Warning:</b> the result takes <i>O(n log n)</i> space, so use cautiously.
   *
   * <p>This uses an efficient binary recursive algorithm to compute the factorial with balanced
   * multiplies. It also removes all the 2s from the intermediate products (shifting them back in at
   * the end).
   *
   * @throws IllegalArgumentException if {@code n < 0}
   */
  public static BigInteger factorial(int n) {
    checkNonNegative("n", n);

    // If the factorial is small enough, just use LongMath to do it.
    if (n < LongMath.factorials.length) {
      return BigInteger.valueOf(LongMath.factorials[n]);
    }

    // Pre-allocate space for our list of intermediate BigIntegers.
    int approxSize = IntMath.divide(n * IntMath.log2(n, CEILING), Long.SIZE, CEILING);
    ArrayList<BigInteger> bignums = new ArrayList<>(approxSize);
