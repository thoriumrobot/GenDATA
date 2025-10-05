// Source-based slice around line 2496
// Method: com.google.common.collect.MapMakerInternalMap.keySet

  }

  @Override
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
