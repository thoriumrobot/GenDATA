// Source-based slice around line 592
// Method: <com.google.common.collect.ImmutableMultimap: Set createKeySet()>

   * Returns an immutable set of the distinct keys in this multimap, in the same order as they
   * appear in this multimap.
   */
  @Override
  public ImmutableSet<K> keySet() {
    return map.keySet();
  }

  @Override
  Set<K> createKeySet() {
    throw new AssertionError("unreachable");
  }

  /**
   * Returns an immutable map that associates each key with its corresponding values in the
   * multimap. Keys and values appear in the same order as in this multimap.
   */
  @Override
  @SuppressWarnings("unchecked") // a widening cast
  public ImmutableMap<K, Collection<V>> asMap() {
