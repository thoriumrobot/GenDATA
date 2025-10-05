// Source-based slice around line 632
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder setKeyStrength(Strength)>

   * @throws IllegalStateException if the key strength was already set
   */
  @GwtIncompatible // java.lang.ref.WeakReference
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> weakKeys() {
    return setKeyStrength(Strength.WEAK);
  }

  @CanIgnoreReturnValue
  CacheBuilder<K, V> setKeyStrength(Strength strength) {
    checkState(keyStrength == null, "Key strength was already set to %s", keyStrength);
    keyStrength = checkNotNull(strength);
    return this;
  }

  Strength getKeyStrength() {
    return MoreObjects.firstNonNull(keyStrength, Strength.STRONG);
  }

  /**
