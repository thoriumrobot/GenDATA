// Source-based slice around line 613
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableCollection entries()>

  }

  @Override
  Map<K, Collection<V>> createAsMap() {
    throw new AssertionError("should never be called");
  }

  /** Returns an immutable collection of all key-value pairs in the multimap. */
  @Override
  public ImmutableCollection<Entry<K, V>> entries() {
    return (ImmutableCollection<Entry<K, V>>) super.entries();
  }

  @Override
  ImmutableCollection<Entry<K, V>> createEntries() {
    return new EntryCollection<>(this);
  }

  private static final class EntryCollection<K, V> extends ImmutableCollection<Entry<K, V>> {
    @Weak final ImmutableMultimap<K, V> multimap;
