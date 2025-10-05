// Source-based slice around line 98
// Method: com.google.common.collect.SingletonImmutableBiMap.lazyInverse

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
    } else {
      // racy single-check idiom
      ImmutableBiMap<V, K> result = lazyInverse;
      if (result == null) {
        return lazyInverse = new SingletonImmutableBiMap<>(singleValue, singleKey, this);
