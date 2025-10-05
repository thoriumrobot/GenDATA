// Source-based slice around line 199
// Method: <com.google.common.primitives.UnsignedLongs: void sort(long[],int,int)>

    sort(array, 0, array.length);
  }

  /**
   * Sorts the array between {@code fromIndex} inclusive and {@code toIndex} exclusive, treating its
   * elements as unsigned 64-bit integers.
   *
   * @since 23.1
   */
  public static void sort(long[] array, int fromIndex, int toIndex) {
    checkNotNull(array);
    checkPositionIndexes(fromIndex, toIndex, array.length);
    for (int i = fromIndex; i < toIndex; i++) {
      array[i] = flip(array[i]);
    }
    Arrays.sort(array, fromIndex, toIndex);
    for (int i = fromIndex; i < toIndex; i++) {
      array[i] = flip(array[i]);
    }
  }
