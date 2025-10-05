// Source-based slice around line 377
// Method: <com.google.common.primitives.ImmutableIntArray: int get(int)>

    return end == start;
  }

  /**
   * Returns the {@code int} value present at the given index.
   *
   * @throws IndexOutOfBoundsException if {@code index} is negative, or greater than or equal to
   *     {@link #length}
   */
  public int get(int index) {
    Preconditions.checkElementIndex(index, length());
    return array[start + index];
  }

  /**
   * Returns the smallest index for which {@link #get} returns {@code target}, or {@code -1} if no
   * such index exists. Equivalent to {@code asList().indexOf(target)}.
   */
  public int indexOf(int target) {
    for (int i = start; i < end; i++) {
