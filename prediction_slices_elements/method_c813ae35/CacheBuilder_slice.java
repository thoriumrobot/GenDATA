// Source-based slice around line 661
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder weakValues()>

   * <p>Entries with values that have been garbage collected may be counted in {@link Cache#size},
   * but will never be visible to read or write operations; such entries are cleaned up as part of
   * the routine maintenance described in the class javadoc.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalStateException if the value strength was already set
   */
  @GwtIncompatible // java.lang.ref.WeakReference
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> weakValues() {
    return setValueStrength(Strength.WEAK);
  }

  /**
   * Specifies that each value (not key) stored in the cache should be wrapped in a {@link
   * SoftReference} (by default, strong references are used). Softly-referenced objects will be
   * garbage-collected in a <i>globally</i> least-recently-used manner, in response to memory
   * demand.
   *
   * <p><b>Warning:</b> in most circumstances it is better to set a per-cache {@linkplain
