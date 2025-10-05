// Source-based slice around line 277
// Method: <com.google.common.primitives.Doubles: double[] concat(double[])>

   * Returns the values from each provided array combined into a single array. For example, {@code
   * concat(new double[] {a, b}, new double[] {}, new double[] {c}} returns the array {@code {a, b,
   * c}}.
   *
   * @param arrays zero or more {@code double} arrays
   * @return a single array containing all the values from the source arrays, in order
   * @throws IllegalArgumentException if the total number of elements in {@code arrays} does not fit
   *     in an {@code int}
   */
  public static double[] concat(double[]... arrays) {
    long length = 0;
    for (double[] array : arrays) {
      length += array.length;
    }
    double[] result = new double[checkNoOverflow(length)];
    int pos = 0;
    for (double[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
