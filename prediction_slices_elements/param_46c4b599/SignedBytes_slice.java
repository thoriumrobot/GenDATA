// Source-based slice around line 211
// Method: <com.google.common.primitives.SignedBytes: void sortDescending(byte[],int,int)>

    sortDescending(array, 0, array.length);
  }

  /**
   * Sorts the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive in descending order.
   *
   * @since 23.1
   */
  public static void sortDescending(byte[] array, int fromIndex, int toIndex) {
    checkNotNull(array);
    checkPositionIndexes(fromIndex, toIndex, array.length);
    Arrays.sort(array, fromIndex, toIndex);
    Bytes.reverse(array, fromIndex, toIndex);
  }
}
