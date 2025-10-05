// Source-based slice around line 257
// Method: <com.google.common.primitives.Doubles: double constrainToRange(double,double,double)>

   *
   * <p><b>Java 21+ users:</b> Use {@code Math.clamp} instead.
   *
   * @param value the {@code double} value to constrain
   * @param min the lower bound (inclusive) of the range to constrain {@code value} to
   * @param max the upper bound (inclusive) of the range to constrain {@code value} to
   * @throws IllegalArgumentException if {@code min > max}
   * @since 21.0
   */
  public static double constrainToRange(double value, double min, double max) {
    // avoid auto-boxing by not using Preconditions.checkArgument(); see Guava issue 3984
    // Reject NaN by testing for the good case (min <= max) instead of the bad (min > max).
    if (min <= max) {
      return Math.min(Math.max(value, min), max);
    }
    throw new IllegalArgumentException(
        lenientFormat("min (%s) must be less than or equal to max (%s)", min, max));
  }

  /**
