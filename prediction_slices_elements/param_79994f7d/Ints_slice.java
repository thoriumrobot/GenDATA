// Source-based slice around line 132
// Method: <com.google.common.primitives.Ints: int compare(int,int)>

   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Integer#compare} method instead.
   *
   * @param a the first {@code int} to compare
   * @param b the second {@code int} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  @InlineMe(replacement = "Integer.compare(a, b)")
  public static int compare(int a, int b) {
    return Integer.compare(a, b);
  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * @param array an array of {@code int} values, possibly empty
   * @param target a primitive {@code int} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
