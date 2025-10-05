// Source-based slice around line 266
// Method: <com.google.common.primitives.Chars: char constrainToRange(char,char,char)>

   * unchanged. If {@code value} is less than {@code min}, {@code min} is returned, and if {@code
   * value} is greater than {@code max}, {@code max} is returned.
   *
   * @param value the {@code char} value to constrain
   * @param min the lower bound (inclusive) of the range to constrain {@code value} to
   * @param max the upper bound (inclusive) of the range to constrain {@code value} to
   * @throws IllegalArgumentException if {@code min > max}
   * @since 21.0
   */
  public static char constrainToRange(char value, char min, char max) {
    checkArgument(min <= max, "min (%s) must be less than or equal to max (%s)", min, max);
    return value < min ? min : value < max ? value : max;
  }

  /**
   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new char[] {a, b}, new char[] {}, new char[] {c}} returns the array {@code {a, b, c}}.
   *
   * @param arrays zero or more {@code char} arrays
   * @return a single array containing all the values from the source arrays, in order
