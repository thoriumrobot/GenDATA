// Source-based slice around line 90
// Method: <com.google.common.primitives.UnsignedLongs: long min(long)>


  /**
   * Returns the least value present in {@code array}, treating values as unsigned.
   *
   * @param array a <i>nonempty</i> array of unsigned {@code long} values
   * @return the value present in {@code array} that is less than or equal to every other value in
   *     the array according to {@link #compare}
   * @throws IllegalArgumentException if {@code array} is empty
   */
  public static long min(long... array) {
    checkArgument(array.length > 0);
    long min = flip(array[0]);
    for (int i = 1; i < array.length; i++) {
      long next = flip(array[i]);
      if (next < min) {
        min = next;
      }
    }
    return flip(min);
  }
