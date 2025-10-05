// Source-based slice around line 451
// Method: <com.google.common.primitives.Floats: void reverse(float[])>

    reverse(array, fromIndex, toIndex);
  }

  /**
   * Reverses the elements of {@code array}. This is equivalent to {@code
   * Collections.reverse(Floats.asList(array))}, but is likely to be more efficient.
   *
   * @since 23.1
   */
  public static void reverse(float[] array) {
    checkNotNull(array);
    reverse(array, 0, array.length);
  }

  /**
   * Reverses the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive. This is equivalent to {@code
   * Collections.reverse(Floats.asList(array).subList(fromIndex, toIndex))}, but is likely to be
   * more efficient.
   *
