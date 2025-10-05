// Source-based slice around line 285
// Method: <com.google.common.primitives.Ints: int constrainToRange(int,int,int)>

   *
   * @param value the {@code int} value to constrain
   * @param min the lower bound (inclusive) of the range to constrain {@code value} to
   * @param max the upper bound (inclusive) of the range to constrain {@code value} to
   * @throws IllegalArgumentException if {@code min > max}
   * @since 21.0
   */
  // A call to bare "min" or "max" would resolve to our varargs method, not to any static import.
  @SuppressWarnings("StaticImportPreferred")
  public static int constrainToRange(int value, int min, int max) {
    checkArgument(min <= max, "min (%s) must be less than or equal to max (%s)", min, max);
    return Math.min(Math.max(value, min), max);
  }

  /**
   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new int[] {a, b}, new int[] {}, new int[] {c}} returns the array {@code {a, b, c}}.
   *
   * @param arrays zero or more {@code int} arrays
   * @return a single array containing all the values from the source arrays, in order
