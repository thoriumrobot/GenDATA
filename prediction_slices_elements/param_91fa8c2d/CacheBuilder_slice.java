// Source-based slice around line 368
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder keyEquivalence(Equivalence)>

   * Sets a custom {@code Equivalence} strategy for comparing keys.
   *
   * <p>By default, the cache uses {@link Equivalence#identity} to determine key equality when
   * {@link #weakKeys} is specified, and {@link Equivalence#equals()} otherwise.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
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
