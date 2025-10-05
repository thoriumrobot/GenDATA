// Source-based slice around line 687
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder softValues()>

   * <p>Entries with values that have been garbage collected may be counted in {@link Cache#size},
   * but will never be visible to read or write operations; such entries are cleaned up as part of
   * the routine maintenance described in the class javadoc.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalStateException if the value strength was already set
   */
  @GwtIncompatible // java.lang.ref.SoftReference
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> softValues() {
    return setValueStrength(Strength.SOFT);
  }

  @CanIgnoreReturnValue
  CacheBuilder<K, V> setValueStrength(Strength strength) {
    checkState(valueStrength == null, "Value strength was already set to %s", valueStrength);
    valueStrength = checkNotNull(strength);
    return this;
  }

