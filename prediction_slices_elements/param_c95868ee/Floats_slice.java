// Source-based slice around line 88
// Method: <com.google.common.primitives.Floats: int compare(float,float)>

   * <p><b>Note:</b> this method simply delegates to the JDK method {@link Float#compare}. It is
   * provided for consistency with the other primitive types, whose compare methods were not added
   * to the JDK until JDK 7.
   *
   * @param a the first {@code float} to compare
   * @param b the second {@code float} to compare
   * @return the result of invoking {@link Float#compare(float, float)}
   */
  @InlineMe(replacement = "Float.compare(a, b)")
  public static int compare(float a, float b) {
    return Float.compare(a, b);
  }

  /**
   * Returns {@code true} if {@code value} represents a real number. This is equivalent to, but not
   * necessarily implemented as, {@code !(Float.isInfinite(value) || Float.isNaN(value))}.
   *
   * <p>Prefer {@link Float#isFinite(float)} instead.
   *
   * @since 10.0
