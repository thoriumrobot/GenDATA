// Source-based slice around line 253
// Method: <com.google.common.primitives.UnsignedInts: void sortDescending(int[])>

    }
  }

  /**
   * Sorts the elements of {@code array} in descending order, interpreting them as unsigned 32-bit
   * integers.
   *
   * @since 23.1
   */
  public static void sortDescending(int[] array) {
    checkNotNull(array);
    sortDescending(array, 0, array.length);
  }

  /**
   * Sorts the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive in descending order, interpreting them as unsigned 32-bit integers.
   *
   * @since 23.1
   */
