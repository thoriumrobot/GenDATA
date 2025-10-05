// Source-based slice around line 405
// Method: <com.google.common.primitives.Bytes: void reverse(byte[],int,int)>

   * Reverses the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive. This is equivalent to {@code
   * Collections.reverse(Bytes.asList(array).subList(fromIndex, toIndex))}, but is likely to be more
   * efficient.
   *
   * @throws IndexOutOfBoundsException if {@code fromIndex < 0}, {@code toIndex > array.length}, or
   *     {@code toIndex > fromIndex}
   * @since 23.1
   */
  public static void reverse(byte[] array, int fromIndex, int toIndex) {
    checkNotNull(array);
    checkPositionIndexes(fromIndex, toIndex, array.length);
    for (int i = fromIndex, j = toIndex - 1; i < j; i++, j--) {
      byte tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
    }
  }

  /**
