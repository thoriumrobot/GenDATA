// Source-based slice around line 379
// Method: <com.google.common.math.LongMath: long divide(long,long,RoundingMode)>

   * RoundingMode}. If the {@code RoundingMode} is {@link RoundingMode#DOWN}, then this method is
   * equivalent to regular Java division, {@code p / q}; and if it is {@link RoundingMode#FLOOR},
   * then this method is equivalent to {@link Math#floorDiv(long,long) Math.floorDiv}{@code (p, q)}.
   *
   * @throws ArithmeticException if {@code q == 0}, or if {@code mode == UNNECESSARY} and {@code a}
   *     is not an integer multiple of {@code b}
   */
  @GwtIncompatible // TODO
  @SuppressWarnings("fallthrough")
  public static long divide(long p, long q, RoundingMode mode) {
    checkNotNull(mode);
    long div = p / q; // throws if q == 0
    long rem = p - q * div; // equals p % q

    if (rem == 0) {
      return div;
    }

    /*
     * Normal Java division rounds towards 0, consistently with RoundingMode.DOWN. We just have to
