// Source-based slice around line 424
// Method: <com.google.common.primitives.Floats: void sortDescending(float[])>


  /**
   * Sorts the elements of {@code array} in descending order.
   *
   * <p>Note that this method uses the total order imposed by {@link Float#compare}, which treats
   * all NaN values as equal and 0.0 as greater than -0.0.
   *
   * @since 23.1
   */
  public static void sortDescending(float[] array) {
    checkNotNull(array);
    sortDescending(array, 0, array.length);
  }

  /**
   * Sorts the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive in descending order.
   *
   * <p>Note that this method uses the total order imposed by {@link Float#compare}, which treats
   * all NaN values as equal and 0.0 as greater than -0.0.
