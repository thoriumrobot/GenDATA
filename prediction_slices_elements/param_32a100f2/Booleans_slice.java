// Source-based slice around line 575
// Method: <com.google.common.primitives.Booleans: void rotate(boolean[],int)>

   * Performs a right rotation of {@code array} of "distance" places, so that the first element is
   * moved to index "distance", and the element at index {@code i} ends up at index {@code (distance
   * + i) mod array.length}. This is equivalent to {@code Collections.rotate(Booleans.asList(array),
   * distance)}, but is somewhat faster.
   *
   * <p>The provided "distance" may be negative, which will rotate left.
   *
   * @since 32.0.0
   */
  public static void rotate(boolean[] array, int distance) {
    rotate(array, distance, 0, array.length);
  }

  /**
   * Performs a right rotation of {@code array} between {@code fromIndex} inclusive and {@code
   * toIndex} exclusive. This is equivalent to {@code
   * Collections.rotate(Booleans.asList(array).subList(fromIndex, toIndex), distance)}, but is
   * somewhat faster.
   *
   * <p>The provided "distance" may be negative, which will rotate left.
