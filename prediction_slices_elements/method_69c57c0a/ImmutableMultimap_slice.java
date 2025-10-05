// Source-based slice around line 487
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableCollection get(K)>

    throw new UnsupportedOperationException();
  }

  /**
   * Returns an immutable collection of the values for the given key. If no mappings in the multimap
   * have the provided key, an empty immutable collection is returned. The values are in the same
   * order as the parameters used to build this multimap.
   */
  @Override
  public abstract ImmutableCollection<V> get(K key);

  /**
   * Returns an immutable multimap which is the inverse of this one. For every key-value mapping in
   * the original, the result will have a mapping with key and value reversed.
   *
   * @since 11.0
   */
  public abstract ImmutableMultimap<V, K> inverse();

  /**
