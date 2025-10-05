// Source-based slice around line 396
// Method: <com.google.common.cache.CacheBuilder: Equivalence getValueEquivalence()>

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

  /**
   * Sets the minimum total size for the internal hash tables. For example, if the initial capacity
   * is {@code 60}, and the concurrency level is {@code 8}, then eight segments are created, each
   * having a hash table of size eight. Providing a large enough estimate at construction time
   * avoids the need for expensive resizing operations later, but setting this value unnecessarily
   * high wastes memory.
   *
