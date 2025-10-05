// Source-based slice around line 692
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder setValueStrength(Strength)>

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

  Strength getValueStrength() {
    return MoreObjects.firstNonNull(valueStrength, Strength.STRONG);
  }

  /**
