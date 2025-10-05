// Source-based slice around line 701
// Method: <com.google.common.math.Quantiles: void swap(double[],int,int)>

    // Now pick the closest of the two candidates. Note that there is no rounding here.
    if (from + to - allRequired[low] - allRequired[high] > 0) {
      return high;
    } else {
      return low;
    }
  }

  /** Swaps the values at {@code i} and {@code j} in {@code array}. */
  private static void swap(double[] array, int i, int j) {
    double temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}
