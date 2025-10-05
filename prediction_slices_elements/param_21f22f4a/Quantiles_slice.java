// Source-based slice around line 603
// Method: <com.google.common.math.Quantiles: void movePivotToStartOfSlice(double[],int,int)>

    return partitionPoint;
  }

  /**
   * Selects the pivot to use, namely the median of the values at {@code from}, {@code to}, and
   * halfway between the two (rounded down), from {@code array}, and ensure (by swapping elements if
   * necessary) that that pivot value appears at the start of the slice i.e. at {@code from}.
   * Expects that {@code from} is strictly less than {@code to}.
   */
  private static void movePivotToStartOfSlice(double[] array, int from, int to) {
    int mid = (from + to) >>> 1;
    // We want to make a swap such that either array[to] <= array[from] <= array[mid], or
    // array[mid] <= array[from] <= array[to]. We know that from < to, so we know mid < to
    // (although it's possible that mid == from, if to == from + 1). Note that the postcondition
    // would be impossible to fulfil if mid == to unless we also have array[from] == array[to].
    boolean toLessThanMid = (array[to] < array[mid]);
    boolean midLessThanFrom = (array[mid] < array[from]);
    boolean toLessThanFrom = (array[to] < array[from]);
    if (toLessThanMid == midLessThanFrom) {
      // Either array[to] < array[mid] < array[from] or array[from] <= array[mid] <= array[to].
