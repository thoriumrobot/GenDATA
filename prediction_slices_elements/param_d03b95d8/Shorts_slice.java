// Source-based slice around line 289
// Method: <com.google.common.primitives.Shorts: short[] concat(short[])>

   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new short[] {a, b}, new short[] {}, new short[] {c}} returns the array {@code {a, b,
   * c}}.
   *
   * @param arrays zero or more {@code short} arrays
   * @return a single array containing all the values from the source arrays, in order
   * @throws IllegalArgumentException if the total number of elements in {@code arrays} does not fit
   *     in an {@code int}
   */
  public static short[] concat(short[]... arrays) {
    long length = 0;
    for (short[] array : arrays) {
      length += array.length;
    }
    short[] result = new short[checkNoOverflow(length)];
    int pos = 0;
    for (short[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
