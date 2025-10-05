// Source-based slice around line 389
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder valueEquivalence(Equivalence)>

   *
   * <p>By default, the cache uses {@link Equivalence#identity} to determine value equality when
   * {@link #weakValues} or {@link #softValues} is specified, and {@link Equivalence#equals()}
   * otherwise.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
   */
  @GwtIncompatible // To be supported
  @CanIgnoreReturnValue
  CacheBuilder<K, V> valueEquivalence(Equivalence<Object> equivalence) {
    checkState(
        valueEquivalence == null, "value equivalence was already set to %s", valueEquivalence);
    this.valueEquivalence = checkNotNull(equivalence);
    return this;
  }

  Equivalence<Object> getValueEquivalence() {
    return MoreObjects.firstNonNull(valueEquivalence, getValueStrength().defaultEquivalence());
  }

