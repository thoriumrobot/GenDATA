// Source-based slice around line 104
// Method: <com.google.common.primitives.Doubles: boolean isFinite(double)>

  /**
   * Returns {@code true} if {@code value} represents a real number. This is equivalent to, but not
   * necessarily implemented as, {@code !(Double.isInfinite(value) || Double.isNaN(value))}.
   *
   * <p>Prefer {@link Double#isFinite(double)} instead.
   *
   * @since 10.0
   */
  @InlineMe(replacement = "Double.isFinite(value)")
  public static boolean isFinite(double value) {
    return Double.isFinite(value);
  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}. Note
   * that this always returns {@code false} when {@code target} is {@code NaN}.
   *
   * @param array an array of {@code double} values, possibly empty
   * @param target a primitive {@code double} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
