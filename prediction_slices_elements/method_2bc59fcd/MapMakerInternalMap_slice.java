// Source-based slice around line 2499
// Method: <com.google.common.collect.MapMakerInternalMap: Set keySet()>

  public void clear() {
    for (Segment<K, V, E, S> segment : segments) {
      segment.clear();
    }
  }

  @LazyInit transient @Nullable Set<K> keySet;

  @Override
  public Set<K> keySet() {
    Set<K> ks = keySet;
    return (ks != null) ? ks : (keySet = new KeySet());
  }

  @LazyInit transient @Nullable Collection<V> values;

  @Override
  public Collection<V> values() {
    Collection<V> vs = values;
    return (vs != null) ? vs : (values = new Values());
