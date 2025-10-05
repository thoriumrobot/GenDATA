// Source-based slice around line 91
// Method: <com.google.common.primitives.SignedBytes: int compare(byte,byte)>

   * that of {@code ((Byte) a).compareTo(b)}.
   *
   * <p><b>Note:</b> this method behaves identically to {@link Byte#compare}.
   *
   * @param a the first {@code byte} to compare
   * @param b the second {@code byte} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  public static int compare(byte a, byte b) {
    return Byte.compare(a, b);
  }

  /**
   * Returns the least value present in {@code array}.
   *
   * @param array a <i>nonempty</i> array of {@code byte} values
   * @return the value present in {@code array} that is less than or equal to every other value in
   *     the array
   * @throws IllegalArgumentException if {@code array} is empty
