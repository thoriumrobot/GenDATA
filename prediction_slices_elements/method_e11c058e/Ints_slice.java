// Source-based slice around line 489
// Method: <com.google.common.primitives.Ints: void sortDescending(int[])>

      return "Ints.lexicographicalComparator()";
    }
  }

  /**
   * Sorts the elements of {@code array} in descending order.
   *
   * @since 23.1
   */
  public static void sortDescending(int[] array) {
    checkNotNull(array);
    sortDescending(array, 0, array.length);
  }

  /**
   * Sorts the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive in descending order.
   *
   * @since 23.1
   */
