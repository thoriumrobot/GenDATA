// Source-based slice around line 78
// Method: <com.google.common.primitives.Shorts: int hashCode(short)>

   * Returns a hash code for {@code value}; obsolete alternative to {@link Short#hashCode(short)}.
   *
   * @param value a primitive {@code short} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Short.hashCode(value)")
  @InlineMeValidationDisabled(
      "The hash code of a short is the int version of the short itself, so it's simplest to return"
          + " that.")
  public static int hashCode(short value) {
    return value;
  }

  /**
   * Returns the {@code short} value that is equal to {@code value}, if possible.
   *
   * @param value any value in the range of the {@code short} type
   * @return the {@code short} value that equals {@code value}
   * @throws IllegalArgumentException if {@code value} is greater than {@link Short#MAX_VALUE} or
   *     less than {@link Short#MIN_VALUE}
