// Source-based slice around line 441
// Method: <com.google.common.primitives.Doubles: void sortDescending(double[],int,int)>

  /**
   * Sorts the elements of {@code array} between {@code fromIndex} inclusive and {@code toIndex}
   * exclusive in descending order.
   *
   * <p>Note that this method uses the total order imposed by {@link Double#compare}, which treats
   * all NaN values as equal and 0.0 as greater than -0.0.
   *
   * @since 23.1
   */
  public static void sortDescending(double[] array, int fromIndex, int toIndex) {
    checkNotNull(array);
    checkPositionIndexes(fromIndex, toIndex, array.length);
    Arrays.sort(array, fromIndex, toIndex);
    reverse(array, fromIndex, toIndex);
  }

  /**
   * Reverses the elements of {@code array}. This is equivalent to {@code
   * Collections.reverse(Doubles.asList(array))}, but is likely to be more efficient.
   *
