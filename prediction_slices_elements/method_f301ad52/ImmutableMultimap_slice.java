// Source-based slice around line 477
// Method: <com.google.common.collect.ImmutableMultimap: void clear()>

  /**
   * Guaranteed to throw an exception and leave the multimap unmodified.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Unsupported operation.
   */
  @Deprecated
  @Override
  @DoNotCall("Always throws UnsupportedOperationException")
  public final void clear() {
    throw new UnsupportedOperationException();
  }

  /**
   * Returns an immutable collection of the values for the given key. If no mappings in the multimap
   * have the provided key, an empty immutable collection is returned. The values are in the same
   * order as the parameters used to build this multimap.
   */
  @Override
  public abstract ImmutableCollection<V> get(K key);
