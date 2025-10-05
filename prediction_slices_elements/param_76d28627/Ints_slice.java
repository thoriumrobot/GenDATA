// Source-based slice around line 299
// Method: <com.google.common.primitives.Ints: int[] concat(int[])>

  /**
   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new int[] {a, b}, new int[] {}, new int[] {c}} returns the array {@code {a, b, c}}.
   *
   * @param arrays zero or more {@code int} arrays
   * @return a single array containing all the values from the source arrays, in order
   * @throws IllegalArgumentException if the total number of elements in {@code arrays} does not fit
   *     in an {@code int}
   */
  public static int[] concat(int[]... arrays) {
    long length = 0;
    for (int[] array : arrays) {
      length += array.length;
    }
    int[] result = new int[checkNoOverflow(length)];
    int pos = 0;
    for (int[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
