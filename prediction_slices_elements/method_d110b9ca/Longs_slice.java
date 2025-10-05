// Source-based slice around line 93
// Method: <com.google.common.primitives.Longs: int compare(long,long)>

   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Long#compare} method instead.
   *
   * @param a the first {@code long} to compare
   * @param b the second {@code long} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  @InlineMe(replacement = "Long.compare(a, b)")
  public static int compare(long a, long b) {
    return Long.compare(a, b);
  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * @param array an array of {@code long} values, possibly empty
   * @param target a primitive {@code long} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
