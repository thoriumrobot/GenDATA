// Source-based slice around line 126
// Method: <com.google.common.primitives.Shorts: int compare(short,short)>

   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Short#compare} method instead.
   *
   * @param a the first {@code short} to compare
   * @param b the second {@code short} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  @InlineMe(replacement = "Short.compare(a, b)")
  public static int compare(short a, short b) {
    return Short.compare(a, b);
  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * @param array an array of {@code short} values, possibly empty
   * @param target a primitive {@code short} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
