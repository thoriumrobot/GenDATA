// Source-based slice around line 163
// Method: <com.google.common.primitives.UnsignedBytes: byte max(byte)>


  /**
   * Returns the greatest value present in {@code array}, treating values as unsigned.
   *
   * @param array a <i>nonempty</i> array of {@code byte} values
   * @return the value present in {@code array} that is greater than or equal to every other value
   *     in the array according to {@link #compare}
   * @throws IllegalArgumentException if {@code array} is empty
   */
  public static byte max(byte... array) {
    checkArgument(array.length > 0);
    int max = toUnsignedInt(array[0]);
    for (int i = 1; i < array.length; i++) {
      int next = toUnsignedInt(array[i]);
      if (next > max) {
        max = next;
      }
    }
    return (byte) max;
  }
