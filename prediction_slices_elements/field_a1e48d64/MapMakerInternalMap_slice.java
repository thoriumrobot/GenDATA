// Source-based slice around line 2504
// Method: com.google.common.collect.MapMakerInternalMap.values


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
  }

  @LazyInit transient @Nullable Set<Entry<K, V>> entrySet;

  @Override
