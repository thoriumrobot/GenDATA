// Source-based slice around line 380
// Method: <com.google.common.primitives.ImmutableDoubleArray: double get(int)>

    return end == start;
  }

  /**
   * Returns the {@code double} value present at the given index.
   *
   * @throws IndexOutOfBoundsException if {@code index} is negative, or greater than or equal to
   *     {@link #length}
   */
  public double get(int index) {
    Preconditions.checkElementIndex(index, length());
    return array[start + index];
  }

  /**
   * Returns the smallest index for which {@link #get} returns {@code target}, or {@code -1} if no
   * such index exists. Values are compared as if by {@link Double#equals}. Equivalent to {@code
   * asList().indexOf(target)}.
   */
  public int indexOf(double target) {
