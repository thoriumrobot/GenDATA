  public int m(int[] a, @IntRange(from = 0, to = 12) int i) {
    // :: error: (array.access.unsafe.high.range)
    return a[i];
  }
