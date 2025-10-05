// Source-based slice around line 217
// Method: <com.google.common.primitives.UnsignedLongs: void sortDescending(long[])>

    }
  }

  /**
   * Sorts the elements of {@code array} in descending order, interpreting them as unsigned 64-bit
   * integers.
   *
   * @since 23.1
   */
  public static void sortDescending(long[] array) {
    checkNotNull(array);
    sortDescending(array, 0, array.length);
  }

  /**
   * Sorts the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive in descending order, interpreting them as unsigned 64-bit integers.
   *
   * @since 23.1
   */
