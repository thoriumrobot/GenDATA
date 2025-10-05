// Source-based slice around line 213
// Method: <com.google.common.primitives.Longs: long max(long)>


  /**
   * Returns the greatest value present in {@code array}.
   *
   * @param array a <i>nonempty</i> array of {@code long} values
   * @return the value present in {@code array} that is greater than or equal to every other value
   *     in the array
   * @throws IllegalArgumentException if {@code array} is empty
   */
  public static long max(long... array) {
    checkArgument(array.length > 0);
    long max = array[0];
    for (int i = 1; i < array.length; i++) {
      if (array[i] > max) {
        max = array[i];
      }
    }
    return max;
  }

