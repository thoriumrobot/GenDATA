// Source-based slice around line 364
// Method: <com.google.common.primitives.ImmutableLongArray: int length()>

  }

  private ImmutableLongArray(long[] array, int start, int end) {
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
   * Returns the {@code long} value present at the given index.
