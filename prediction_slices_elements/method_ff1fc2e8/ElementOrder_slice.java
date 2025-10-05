// Source-based slice around line 162
// Method: <com.google.common.graph.ElementOrder: boolean equals(Object)>

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
    }
    if (!(obj instanceof ElementOrder)) {
      return false;
    }

    ElementOrder<?> other = (ElementOrder<?>) obj;
    return (type == other.type) && Objects.equals(comparator, other.comparator);
  }
