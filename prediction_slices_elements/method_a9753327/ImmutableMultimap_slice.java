// Source-based slice around line 602
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableMap asMap()>

    throw new AssertionError("unreachable");
  }

  /**
   * Returns an immutable map that associates each key with its corresponding values in the
   * multimap. Keys and values appear in the same order as in this multimap.
   */
  @Override
  @SuppressWarnings("unchecked") // a widening cast
  public ImmutableMap<K, Collection<V>> asMap() {
    return (ImmutableMap) map;
  }

  @Override
  Map<K, Collection<V>> createAsMap() {
    throw new AssertionError("should never be called");
  }

  /** Returns an immutable collection of all key-value pairs in the multimap. */
  @Override
