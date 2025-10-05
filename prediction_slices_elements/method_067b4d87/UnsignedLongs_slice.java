// Source-based slice around line 110
// Method: <com.google.common.primitives.UnsignedLongs: long max(long)>


  /**
   * Returns the greatest value present in {@code array}, treating values as unsigned.
   *
   * @param array a <i>nonempty</i> array of unsigned {@code long} values
   * @return the value present in {@code array} that is greater than or equal to every other value
   *     in the array according to {@link #compare}
   * @throws IllegalArgumentException if {@code array} is empty
   */
  public static long max(long... array) {
    checkArgument(array.length > 0);
    long max = flip(array[0]);
    for (int i = 1; i < array.length; i++) {
      long next = flip(array[i]);
      if (next > max) {
        max = next;
      }
    }
    return flip(max);
  }
