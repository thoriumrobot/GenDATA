// Source-based slice around line 540
// Method: <com.google.common.primitives.Booleans: void reverse(boolean[])>

    return count;
  }

  /**
   * Reverses the elements of {@code array}. This is equivalent to {@code
   * Collections.reverse(Booleans.asList(array))}, but is likely to be more efficient.
   *
   * @since 23.1
   */
  public static void reverse(boolean[] array) {
    checkNotNull(array);
    reverse(array, 0, array.length);
  }

  /**
   * Reverses the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive. This is equivalent to {@code
   * Collections.reverse(Booleans.asList(array).subList(fromIndex, toIndex))}, but is likely to be
   * more efficient.
   *
