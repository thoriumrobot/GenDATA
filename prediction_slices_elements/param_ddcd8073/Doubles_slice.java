// Source-based slice around line 91
// Method: <com.google.common.primitives.Doubles: int compare(double,double)>

   * provided for consistency with the other primitive types, whose compare methods were not added
   * to the JDK until JDK 7.
   *
   * @param a the first {@code double} to compare
   * @param b the second {@code double} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  @InlineMe(replacement = "Double.compare(a, b)")
  public static int compare(double a, double b) {
    return Double.compare(a, b);
  }

  /**
   * Returns {@code true} if {@code value} represents a real number. This is equivalent to, but not
   * necessarily implemented as, {@code !(Double.isInfinite(value) || Double.isNaN(value))}.
   *
   * <p>Prefer {@link Double#isFinite(double)} instead.
   *
   * @since 10.0
