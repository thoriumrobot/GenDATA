// Source-based slice around line 93
// Method: <com.google.common.collect.SingletonImmutableBiMap: ImmutableSet createKeySet()>

    return false;
  }

  @Override
  ImmutableSet<Entry<K, V>> createEntrySet() {
    return ImmutableSet.of(immutableEntry(singleKey, singleValue));
  }

  @Override
  ImmutableSet<K> createKeySet() {
    return ImmutableSet.of(singleKey);
  }

  private final transient @Nullable ImmutableBiMap<V, K> inverse;
  @LazyInit @RetainedWith private transient @Nullable ImmutableBiMap<V, K> lazyInverse;

  @Override
  public ImmutableBiMap<V, K> inverse() {
    if (inverse != null) {
      return inverse;
