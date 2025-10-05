// Source-based slice around line 379
// Method: <com.google.common.primitives.ImmutableLongArray: long get(int)>

    return end == start;
  }

  /**
   * Returns the {@code long} value present at the given index.
   *
   * @throws IndexOutOfBoundsException if {@code index} is negative, or greater than or equal to
   *     {@link #length}
   */
  public long get(int index) {
    Preconditions.checkElementIndex(index, length());
    return array[start + index];
  }

  /**
   * Returns the smallest index for which {@link #get} returns {@code target}, or {@code -1} if no
   * such index exists. Equivalent to {@code asList().indexOf(target)}.
   */
  public int indexOf(long target) {
    for (int i = start; i < end; i++) {
