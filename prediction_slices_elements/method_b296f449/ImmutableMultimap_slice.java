// Source-based slice around line 587
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableSet keySet()>

  }

  // views

  /**
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
