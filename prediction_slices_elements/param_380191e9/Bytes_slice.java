// Source-based slice around line 166
// Method: <com.google.common.primitives.Bytes: byte[] concat(byte[])>

  /**
   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new byte[] {a, b}, new byte[] {}, new byte[] {c}} returns the array {@code {a, b, c}}.
   *
   * @param arrays zero or more {@code byte} arrays
   * @return a single array containing all the values from the source arrays, in order
   * @throws IllegalArgumentException if the total number of elements in {@code arrays} does not fit
   *     in an {@code int}
   */
  public static byte[] concat(byte[]... arrays) {
    long length = 0;
    for (byte[] array : arrays) {
      length += array.length;
    }
    byte[] result = new byte[checkNoOverflow(length)];
    int pos = 0;
    for (byte[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
