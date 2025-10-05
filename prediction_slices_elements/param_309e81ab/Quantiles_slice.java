// Source-based slice around line 537
// Method: <com.google.common.math.Quantiles: void selectInPlace(int,double[],int,int)>

   *       to}].
   * </ul>
   *
   * This method will reorder the values with indexes in the range [{@code from}, {@code to}] such
   * that all the values with indexes in the range [{@code from}, {@code required}) are less than or
   * equal to the value with index {@code required}, and all the values with indexes in the range
   * ({@code required}, {@code to}] are greater than or equal to that value. Therefore, the value at
   * {@code required} is the value which would appear at that index in the sorted dataset.
   */
  private static void selectInPlace(int required, double[] array, int from, int to) {
    // If we are looking for the least element in the range, we can just do a linear search for it.
    // (We will hit this whenever we are doing quantile interpolation: our first selection finds
    // the lower value, our second one finds the upper value by looking for the next least element.)
    if (required == from) {
      int min = from;
      for (int index = from + 1; index <= to; index++) {
        if (array[min] > array[index]) {
          min = index;
        }
      }
