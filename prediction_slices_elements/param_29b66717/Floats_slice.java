// Source-based slice around line 274
// Method: <com.google.common.primitives.Floats: float[] concat(float[])>

   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new float[] {a, b}, new float[] {}, new float[] {c}} returns the array {@code {a, b,
   * c}}.
   *
   * @param arrays zero or more {@code float} arrays
   * @return a single array containing all the values from the source arrays, in order
   * @throws IllegalArgumentException if the total number of elements in {@code arrays} does not fit
   *     in an {@code int}
   */
  public static float[] concat(float[]... arrays) {
    long length = 0;
    for (float[] array : arrays) {
      length += array.length;
    }
    float[] result = new float[checkNoOverflow(length)];
    int pos = 0;
    for (float[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
