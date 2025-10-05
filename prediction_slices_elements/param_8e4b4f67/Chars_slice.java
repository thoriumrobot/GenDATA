// Source-based slice around line 73
// Method: <com.google.common.primitives.Chars: int hashCode(char)>

   * Character#hashCode(char)}.
   *
   * @param value a primitive {@code char} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Character.hashCode(value)")
  @InlineMeValidationDisabled(
      "The hash code of a char is the int version of the char itself, so it's simplest to return"
          + " that.")
  public static int hashCode(char value) {
    return value;
  }

  /**
   * Returns the {@code char} value that is equal to {@code value}, if possible.
   *
   * @param value any value in the range of the {@code char} type
   * @return the {@code char} value that equals {@code value}
   * @throws IllegalArgumentException if {@code value} is greater than {@link Character#MAX_VALUE}
   *     or less than {@link Character#MIN_VALUE}
