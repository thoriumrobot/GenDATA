// Source-based slice around line 627
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder weakKeys()>

   * <p>Entries with keys that have been garbage collected may be counted in {@link Cache#size}, but
   * will never be visible to read or write operations; such entries are cleaned up as part of the
   * routine maintenance described in the class javadoc.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
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

