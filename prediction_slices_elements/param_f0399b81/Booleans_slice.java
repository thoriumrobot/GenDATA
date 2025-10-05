// Source-based slice around line 236
// Method: <com.google.common.primitives.Booleans: boolean[] concat(boolean[])>

   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new boolean[] {a, b}, new boolean[] {}, new boolean[] {c}} returns the array {@code {a,
   * b, c}}.
   *
   * @param arrays zero or more {@code boolean} arrays
   * @return a single array containing all the values from the source arrays, in order
   * @throws IllegalArgumentException if the total number of elements in {@code arrays} does not fit
   *     in an {@code int}
   */
  public static boolean[] concat(boolean[]... arrays) {
    long length = 0;
    for (boolean[] array : arrays) {
      length += array.length;
    }
    boolean[] result = new boolean[checkNoOverflow(length)];
    int pos = 0;
    for (boolean[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
