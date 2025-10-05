// Source-based slice around line 547
// Method: <com.google.common.primitives.UnsignedBytes: void sortDescending(byte[])>

    }
  }

  /**
   * Sorts the elements of {@code array} in descending order, interpreting them as unsigned 8-bit
   * integers.
   *
   * @since 23.1
   */
  public static void sortDescending(byte[] array) {
    checkNotNull(array);
    sortDescending(array, 0, array.length);
  }

  /**
   * Sorts the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive in descending order, interpreting them as unsigned 8-bit integers.
   *
   * @since 23.1
   */
