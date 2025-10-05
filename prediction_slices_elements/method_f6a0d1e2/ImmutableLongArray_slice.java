// Source-based slice around line 369
// Method: <com.google.common.primitives.ImmutableLongArray: boolean isEmpty()>

    this.end = end;
  }

  /** Returns the number of values in this array. */
  public int length() {
    return end - start;
  }

  /** Returns {@code true} if there are no values in this array ({@link #length} is zero). */
  public boolean isEmpty() {
    return end == start;
  }

  /**
   * Returns the {@code long} value present at the given index.
   *
   * @throws IndexOutOfBoundsException if {@code index} is negative, or greater than or equal to
   *     {@link #length}
   */
  public long get(int index) {
