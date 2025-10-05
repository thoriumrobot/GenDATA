// Source-based slice around line 575
// Method: <com.google.common.math.Quantiles: int partition(double[],int,int)>


  /**
   * Performs a partition operation on the slice of {@code array} with elements in the range [{@code
   * from}, {@code to}]. Uses the median of {@code from}, {@code to}, and the midpoint between them
   * as a pivot. Returns the index which the slice is partitioned around, i.e. if it returns {@code
   * ret} then we know that the values with indexes in [{@code from}, {@code ret}) are less than or
   * equal to the value at {@code ret} and the values with indexes in ({@code ret}, {@code to}] are
   * greater than or equal to that.
   */
  private static int partition(double[] array, int from, int to) {
    // Select a pivot, and move it to the start of the slice i.e. to index from.
    movePivotToStartOfSlice(array, from, to);
    double pivot = array[from];

    // Move all elements with indexes in (from, to] which are greater than the pivot to the end of
    // the array. Keep track of where those elements begin.
    int partitionPoint = to;
    for (int i = to; i > from; i--) {
      if (array[i] > pivot) {
        swap(array, partitionPoint, i);
