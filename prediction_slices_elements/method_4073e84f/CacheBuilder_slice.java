// Source-based slice around line 374
// Method: <com.google.common.cache.CacheBuilder: Equivalence getKeyEquivalence()>

   */
  @GwtIncompatible // To be supported
  @CanIgnoreReturnValue
  CacheBuilder<K, V> keyEquivalence(Equivalence<Object> equivalence) {
    checkState(keyEquivalence == null, "key equivalence was already set to %s", keyEquivalence);
    keyEquivalence = checkNotNull(equivalence);
    return this;
  }

  Equivalence<Object> getKeyEquivalence() {
    return MoreObjects.firstNonNull(keyEquivalence, getKeyStrength().defaultEquivalence());
  }

  /**
   * Sets a custom {@code Equivalence} strategy for comparing values.
   *
   * <p>By default, the cache uses {@link Equivalence#identity} to determine value equality when
   * {@link #weakValues} or {@link #softValues} is specified, and {@link Equivalence#equals()}
   * otherwise.
   *
