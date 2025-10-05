// Source-based slice around line 567
// Method: <com.google.common.primitives.Longs: void sortDescending(long[])>

      return "Longs.lexicographicalComparator()";
    }
  }

  /**
   * Sorts the elements of {@code array} in descending order.
   *
   * @since 23.1
   */
  public static void sortDescending(long[] array) {
    checkNotNull(array);
    sortDescending(array, 0, array.length);
  }

  /**
   * Sorts the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive in descending order.
   *
   * @since 23.1
   */
