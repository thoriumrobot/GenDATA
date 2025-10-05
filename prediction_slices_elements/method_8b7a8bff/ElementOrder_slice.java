// Source-based slice around line 154
// Method: <com.google.common.graph.ElementOrder: Comparator comparator()>

  public Type type() {
    return type;
  }

  /**
   * Returns the {@link Comparator} used.
   *
   * @throws UnsupportedOperationException if comparator is not defined
   */
  public Comparator<T> comparator() {
    if (comparator != null) {
      return comparator;
    }
    throw new UnsupportedOperationException("This ordering does not define a comparator.");
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (obj == this) {
      return true;
