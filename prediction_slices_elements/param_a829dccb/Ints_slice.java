// Source-based slice around line 79
// Method: <com.google.common.primitives.Ints: int hashCode(int)>

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link Integer#hashCode(int)}.
   *
   * @param value a primitive {@code int} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Integer.hashCode(value)")
  @InlineMeValidationDisabled(
      "The hash code of a int is the int itself, so it's simplest to return that.")
  public static int hashCode(int value) {
    return value;
  }

  /**
   * Returns the {@code int} value that is equal to {@code value}, if possible.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated. Use {@link
   * Math#toIntExact(long)} instead, but be aware that that method throws {@link
   * ArithmeticException} rather than {@link IllegalArgumentException}.
   *
