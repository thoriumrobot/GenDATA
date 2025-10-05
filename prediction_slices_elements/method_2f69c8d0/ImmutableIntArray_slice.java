// Source-based slice around line 362
// Method: <com.google.common.primitives.ImmutableIntArray: int length()>

  }

  private ImmutableIntArray(int[] array, int start, int end) {
    this.array = array;
    this.start = start;
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
   * Returns the {@code int} value present at the given index.
