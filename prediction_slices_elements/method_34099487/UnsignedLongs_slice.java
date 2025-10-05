// Source-based slice around line 188
// Method: <com.google.common.primitives.UnsignedLongs: void sort(long[])>

      return "UnsignedLongs.lexicographicalComparator()";
    }
  }

  /**
   * Sorts the array, treating its elements as unsigned 64-bit integers.
   *
   * @since 23.1
   */
  public static void sort(long[] array) {
    checkNotNull(array);
    sort(array, 0, array.length);
  }

  /**
   * Sorts the array between {@code fromIndex} inclusive and {@code toIndex} exclusive, treating its
   * elements as unsigned 64-bit integers.
   *
   * @since 23.1
   */
