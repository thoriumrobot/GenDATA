// Source-based slice around line 2512
// Method: com.google.common.collect.MapMakerInternalMap.entrySet


  @LazyInit transient @Nullable Collection<V> values;

  @Override
  public Collection<V> values() {
    Collection<V> vs = values;
    return (vs != null) ? vs : (values = new Values());
  }

  @LazyInit transient @Nullable Set<Entry<K, V>> entrySet;

  @Override
  public Set<Entry<K, V>> entrySet() {
    Set<Entry<K, V>> es = entrySet;
    return (es != null) ? es : (entrySet = new EntrySet());
  }

  // Iterator Support

  abstract class HashIterator<T> implements Iterator<T> {
