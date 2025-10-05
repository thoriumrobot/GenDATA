// Source-based slice around line 64
// Method: <com.google.common.primitives.Bytes: int hashCode(byte)>

   * Returns a hash code for {@code value}; obsolete alternative to {@link Byte#hashCode(byte)}.
   *
   * @param value a primitive {@code byte} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Byte.hashCode(value)")
  @InlineMeValidationDisabled(
      "The hash code of a byte is the int version of the byte itself, so it's simplest to return"
          + " that.")
  public static int hashCode(byte value) {
    return value;
  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * @param array an array of {@code byte} values, possibly empty
   * @param target a primitive {@code byte} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
