// Source-based slice around line 189
// Method: <com.google.common.graph.ElementOrder: Map createMap(int)>

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
      case UNORDERED:
        return Maps.newHashMapWithExpectedSize(expectedSize);
      case INSERTION:
      case STABLE:
        return Maps.newLinkedHashMapWithExpectedSize(expectedSize);
      case SORTED:
        return Maps.newTreeMap(comparator());
    }
    throw new AssertionError();
