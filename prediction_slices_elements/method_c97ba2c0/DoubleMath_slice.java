// Source-based slice around line 242
// Method: <com.google.common.math.DoubleMath: int log2(double,RoundingMode)>

   *
   * <p>Regardless of the rounding mode, this is faster than {@code (int) log2(x)}.
   *
   * @throws IllegalArgumentException if {@code x <= 0.0}, {@code x} is NaN, or {@code x} is
   *     infinite
   */
  @GwtIncompatible // java.lang.Math.getExponent, com.google.common.math.DoubleUtils
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings({"fallthrough", "ShortCircuitBoolean"})
  public static int log2(double x, RoundingMode mode) {
    checkArgument(x > 0.0 && isFinite(x), "x must be positive and finite");
    int exponent = getExponent(x);
    if (!isNormal(x)) {
      return log2(x * IMPLICIT_BIT, mode) - SIGNIFICAND_BITS;
      // Do the calculation on a normal value.
    }
    // x is positive, finite, and normal
    boolean increment;
    switch (mode) {
      case UNNECESSARY:
