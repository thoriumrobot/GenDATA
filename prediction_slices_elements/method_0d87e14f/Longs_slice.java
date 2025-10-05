// Source-based slice around line 254
// Method: <com.google.common.primitives.Longs: long[] concat(long[])>

  /**
   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new long[] {a, b}, new long[] {}, new long[] {c}} returns the array {@code {a, b, c}}.
   *
   * @param arrays zero or more {@code long} arrays
   * @return a single array containing all the values from the source arrays, in order
   * @throws IllegalArgumentException if the total number of elements in {@code arrays} does not fit
   *     in an {@code int}
   */
  public static long[] concat(long[]... arrays) {
    long length = 0;
    for (long[] array : arrays) {
      length += array.length;
    }
    long[] result = new long[checkNoOverflow(length)];
    int pos = 0;
    for (long[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
