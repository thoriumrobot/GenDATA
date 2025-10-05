// Source-based slice around line 224
// Method: <com.google.common.primitives.UnsignedInts: void sort(int[])>

      return "UnsignedInts.lexicographicalComparator()";
    }
  }

  /**
   * Sorts the array, treating its elements as unsigned 32-bit integers.
   *
   * @since 23.1
   */
  public static void sort(int[] array) {
    checkNotNull(array);
    sort(array, 0, array.length);
  }

  /**
   * Sorts the array between {@code fromIndex} inclusive and {@code toIndex} exclusive, treating its
   * elements as unsigned 32-bit integers.
   *
   * @since 23.1
   */
