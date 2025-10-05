// Source-based slice around line 70
// Method: <com.google.common.primitives.Floats: int hashCode(float)>

  public static final int BYTES = Float.BYTES;

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link Float#hashCode(float)}.
   *
   * @param value a primitive {@code float} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Float.hashCode(value)")
  public static int hashCode(float value) {
    return Float.hashCode(value);
  }

  /**
   * Compares the two specified {@code float} values using {@link Float#compare(float, float)}. You
   * may prefer to invoke that method directly; this method exists only for consistency with the
   * other utilities in this package.
   *
   * <p><b>Note:</b> this method simply delegates to the JDK method {@link Float#compare}. It is
   * provided for consistency with the other primitive types, whose compare methods were not added
