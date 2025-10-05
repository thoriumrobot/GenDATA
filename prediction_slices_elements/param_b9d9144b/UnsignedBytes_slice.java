// Source-based slice around line 558
// Method: <com.google.common.primitives.UnsignedBytes: void sortDescending(byte[],int,int)>

    sortDescending(array, 0, array.length);
  }

  /**
   * Sorts the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive in descending order, interpreting them as unsigned 8-bit integers.
   *
   * @since 23.1
   */
  public static void sortDescending(byte[] array, int fromIndex, int toIndex) {
    checkNotNull(array);
    checkPositionIndexes(fromIndex, toIndex, array.length);
    for (int i = fromIndex; i < toIndex; i++) {
      array[i] ^= Byte.MAX_VALUE;
    }
    Arrays.sort(array, fromIndex, toIndex);
    for (int i = fromIndex; i < toIndex; i++) {
      array[i] ^= Byte.MAX_VALUE;
    }
  }
