// Source-based slice around line 518
// Method: <com.google.common.primitives.UnsignedBytes: void sort(byte[])>

  private static byte flip(byte b) {
    return (byte) (b ^ 0x80);
  }

  /**
   * Sorts the array, treating its elements as unsigned bytes.
   *
   * @since 23.1
   */
  public static void sort(byte[] array) {
    checkNotNull(array);
    sort(array, 0, array.length);
  }

  /**
   * Sorts the array between {@code fromIndex} inclusive and {@code toIndex} exclusive, treating its
   * elements as unsigned bytes.
   *
   * @since 23.1
   */
