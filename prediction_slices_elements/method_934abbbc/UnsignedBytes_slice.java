// Source-based slice around line 143
// Method: <com.google.common.primitives.UnsignedBytes: byte min(byte)>


  /**
   * Returns the least value present in {@code array}, treating values as unsigned.
   *
   * @param array a <i>nonempty</i> array of {@code byte} values
   * @return the value present in {@code array} that is less than or equal to every other value in
   *     the array according to {@link #compare}
   * @throws IllegalArgumentException if {@code array} is empty
   */
  public static byte min(byte... array) {
    checkArgument(array.length > 0);
    int min = toUnsignedInt(array[0]);
    for (int i = 1; i < array.length; i++) {
      int next = toUnsignedInt(array[i]);
      if (next < min) {
        min = next;
      }
    }
    return (byte) min;
  }
