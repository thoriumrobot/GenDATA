// Source-based slice around line 495
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableMultimap inverse()>

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
   * Guaranteed to throw an exception and leave the multimap unmodified.
   *
   * @throws UnsupportedOperationException always
   * @deprecated Unsupported operation.
   */
  @CanIgnoreReturnValue
  @Deprecated
  @Override
