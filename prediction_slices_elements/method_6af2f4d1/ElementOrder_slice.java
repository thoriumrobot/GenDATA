// Source-based slice around line 180
// Method: <com.google.common.graph.ElementOrder: String toString()>

    return (type == other.type) && Objects.equals(comparator, other.comparator);
  }

  @Override
  public int hashCode() {
    return Objects.hash(type, comparator);
  }

  @Override
  public String toString() {
    ToStringHelper helper = MoreObjects.toStringHelper(this).add("type", type);
    if (comparator != null) {
      helper.add("comparator", comparator);
    }
    return helper.toString();
  }

  /** Returns an empty mutable map whose keys will respect this {@link ElementOrder}. */
  <K extends T, V> Map<K, V> createMap(int expectedSize) {
    switch (type) {
