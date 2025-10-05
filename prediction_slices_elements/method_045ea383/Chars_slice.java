// Source-based slice around line 122
// Method: <com.google.common.primitives.Chars: int compare(char,char)>

   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Character#compare} method instead.
   *
   * @param a the first {@code char} to compare
   * @param b the second {@code char} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  @InlineMe(replacement = "Character.compare(a, b)")
  public static int compare(char a, char b) {
    return Character.compare(a, b);
  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * @param array an array of {@code char} values, possibly empty
   * @param target a primitive {@code char} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
