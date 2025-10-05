// Source-based slice around line 222
// Method: <com.google.common.math.IntMath: int pow(int,int)>

   * Returns {@code b} to the {@code k}th power. Even if the result overflows, it will be equal to
   * {@code BigInteger.valueOf(b).pow(k).intValue()}. This implementation runs in {@code O(log k)}
   * time.
   *
   * <p>Compare {@link #checkedPow}, which throws an {@link ArithmeticException} upon overflow.
   *
   * @throws IllegalArgumentException if {@code k < 0}
   */
  @GwtIncompatible // failing tests
  public static int pow(int b, int k) {
    checkNonNegative("exponent", k);
    switch (b) {
      case 0:
        return (k == 0) ? 1 : 0;
      case 1:
        return 1;
      case -1:
        return ((k & 1) == 0) ? 1 : -1;
      case 2:
        return (k < Integer.SIZE) ? (1 << k) : 0;
