// Source-based slice around line 145
// Method: <com.google.common.primitives.UnsignedInts: int max(int)>


  /**
   * Returns the greatest value present in {@code array}, treating values as unsigned.
   *
   * @param array a <i>nonempty</i> array of unsigned {@code int} values
   * @return the value present in {@code array} that is greater than or equal to every other value
   *     in the array according to {@link #compare}
   * @throws IllegalArgumentException if {@code array} is empty
   */
  public static int max(int... array) {
    checkArgument(array.length > 0);
    int max = flip(array[0]);
    for (int i = 1; i < array.length; i++) {
      int next = flip(array[i]);
      if (next > max) {
        max = next;
      }
    }
    return flip(max);
  }
