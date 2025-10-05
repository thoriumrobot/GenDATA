// Source-based slice around line 235
// Method: <com.google.common.primitives.UnsignedInts: void sort(int[],int,int)>

    sort(array, 0, array.length);
  }

  /**
   * Sorts the array between {@code fromIndex} inclusive and {@code toIndex} exclusive, treating its
   * elements as unsigned 32-bit integers.
   *
   * @since 23.1
   */
  public static void sort(int[] array, int fromIndex, int toIndex) {
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
