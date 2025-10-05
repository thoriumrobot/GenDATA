// Source-based slice around line 254
// Method: <com.google.common.primitives.Floats: float constrainToRange(float,float,float)>

   *
   * <p><b>Java 21+ users:</b> Use {@code Math.clamp} instead.
   *
   * @param value the {@code float} value to constrain
   * @param min the lower bound (inclusive) of the range to constrain {@code value} to
   * @param max the upper bound (inclusive) of the range to constrain {@code value} to
   * @throws IllegalArgumentException if {@code min > max}
   * @since 21.0
   */
  public static float constrainToRange(float value, float min, float max) {
    // avoid auto-boxing by not using Preconditions.checkArgument(); see Guava issue 3984
    // Reject NaN by testing for the good case (min <= max) instead of the bad (min > max).
    if (min <= max) {
      return Math.min(Math.max(value, min), max);
    }
    throw new IllegalArgumentException(
        lenientFormat("min (%s) must be less than or equal to max (%s)", min, max));
  }

  /**
