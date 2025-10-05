// Source-based slice around line 280
// Method: <com.google.common.primitives.Chars: char[] concat(char[])>

  /**
   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new char[] {a, b}, new char[] {}, new char[] {c}} returns the array {@code {a, b, c}}.
   *
   * @param arrays zero or more {@code char} arrays
   * @return a single array containing all the values from the source arrays, in order
   * @throws IllegalArgumentException if the total number of elements in {@code arrays} does not fit
   *     in an {@code int}
   */
  public static char[] concat(char[]... arrays) {
    long length = 0;
    for (char[] array : arrays) {
      length += array.length;
    }
    char[] result = new char[checkNoOverflow(length)];
    int pos = 0;
    for (char[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
