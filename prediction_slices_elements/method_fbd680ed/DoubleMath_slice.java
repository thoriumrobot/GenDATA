// Source-based slice around line 363
// Method: <com.google.common.math.DoubleMath: boolean fuzzyEquals(double,double,double)>

   * </ul>
   *
   * <p>This is reflexive and symmetric, but <em>not</em> transitive, so it is <em>not</em> an
   * equivalence relation and <em>not</em> suitable for use in {@link Object#equals}
   * implementations.
   *
   * @throws IllegalArgumentException if {@code tolerance} is {@code < 0} or NaN
   * @since 13.0
   */
  public static boolean fuzzyEquals(double a, double b, double tolerance) {
    MathPreconditions.checkNonNegative("tolerance", tolerance);
    return Math.copySign(a - b, 1.0) <= tolerance
        // copySign(x, 1.0) is a branch-free version of abs(x), but with different NaN semantics
        || (a == b) // needed to ensure that infinities equal themselves
        || (Double.isNaN(a) && Double.isNaN(b));
  }

  /**
   * Compares {@code a} and {@code b} "fuzzily," with a tolerance for nearly-equal values.
   *
