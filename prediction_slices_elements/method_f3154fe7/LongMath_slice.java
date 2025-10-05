// Source-based slice around line 263
// Method: <com.google.common.math.LongMath: long pow(long,int)>


  /**
   * Returns {@code b} to the {@code k}th power. Even if the result overflows, it will be equal to
   * {@code BigInteger.valueOf(b).pow(k).longValue()}. This implementation runs in {@code O(log k)}
   * time.
   *
   * @throws IllegalArgumentException if {@code k < 0}
   */
  @GwtIncompatible // TODO
  public static long pow(long b, int k) {
    checkNonNegative("exponent", k);
    if (-2 <= b && b <= 2) {
      switch ((int) b) {
        case 0:
          return (k == 0) ? 1 : 0;
        case 1:
          return 1;
        case -1:
          return ((k & 1) == 0) ? 1 : -1;
        case 2:
