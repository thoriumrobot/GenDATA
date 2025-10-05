// Source-based slice around line 88
// Method: <com.google.common.collect.SingletonImmutableBiMap: ImmutableSet createEntrySet()>

    return singleValue.equals(value);
  }

  @Override
  boolean isPartialView() {
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
