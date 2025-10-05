// Source-based slice around line 313
// Method: <com.google.common.math.IntMath: int divide(int,int,RoundingMode)>

   * RoundingMode}. If the {@code RoundingMode} is {@link RoundingMode#DOWN}, then this method is
   * equivalent to regular Java division, {@code p / q}; and if it is {@link RoundingMode#FLOOR},
   * then this method is equivalent to {@link Math#floorDiv(int,int) Math.floorDiv}{@code (p, q)}.
   *
   * @throws ArithmeticException if {@code q == 0}, or if {@code mode == UNNECESSARY} and {@code a}
   *     is not an integer multiple of {@code b}
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings({"fallthrough", "ShortCircuitBoolean"})
  public static int divide(int p, int q, RoundingMode mode) {
    checkNotNull(mode);
    if (q == 0) {
      throw new ArithmeticException("/ by zero"); // for GWT
    }
    int div = p / q;
    int rem = p - q * div; // equal to p % q

    if (rem == 0) {
      return div;
    }
