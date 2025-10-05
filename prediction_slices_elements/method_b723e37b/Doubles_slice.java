// Source-based slice around line 72
// Method: <com.google.common.primitives.Doubles: int hashCode(double)>

  public static final int BYTES = Double.BYTES;

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link Double#hashCode(double)}.
   *
   * @param value a primitive {@code double} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Double.hashCode(value)")
  public static int hashCode(double value) {
    return Double.hashCode(value);
  }

  /**
   * Compares the two specified {@code double} values. The sign of the value returned is the same as
   * that of <code>((Double) a).{@linkplain Double#compareTo compareTo}(b)</code>. As with that
   * method, {@code NaN} is treated as greater than all other values, and {@code 0.0 > -0.0}.
   *
   * <p><b>Note:</b> this method simply delegates to the JDK method {@link Double#compare}. It is
   * provided for consistency with the other primitive types, whose compare methods were not added
