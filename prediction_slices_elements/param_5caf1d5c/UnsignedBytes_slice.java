// Source-based slice around line 131
// Method: <com.google.common.primitives.UnsignedBytes: int compare(byte,byte)>

   * Compares the two specified {@code byte} values, treating them as unsigned values between 0 and
   * 255 inclusive. For example, {@code (byte) -127} is considered greater than {@code (byte) 127}
   * because it is seen as having the value of positive {@code 129}.
   *
   * @param a the first {@code byte} to compare
   * @param b the second {@code byte} to compare
   * @return a negative value if {@code a} is less than {@code b}; a positive value if {@code a} is
   *     greater than {@code b}; or zero if they are equal
   */
  public static int compare(byte a, byte b) {
    return toUnsignedInt(a) - toUnsignedInt(b);
  }

  /**
   * Returns the least value present in {@code array}, treating values as unsigned.
   *
   * @param array a <i>nonempty</i> array of {@code byte} values
   * @return the value present in {@code array} that is less than or equal to every other value in
   *     the array according to {@link #compare}
   * @throws IllegalArgumentException if {@code array} is empty
