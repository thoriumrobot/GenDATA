// Source-based slice around line 580
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder weigher(Weigher)>

   * @param weigher the weigher to use in calculating the weight of cache entries
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalStateException if a weigher was already set or {@link #maximumSize(long)} was
   *     previously called
   * @since 11.0
   */
  @GwtIncompatible // To be supported
  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this
  public <K1 extends K, V1 extends V> CacheBuilder<K1, V1> weigher(
      Weigher<? super K1, ? super V1> weigher) {
    checkState(this.weigher == null);
    if (strictParsing) {
      checkState(
          this.maximumSize == UNSET_INT,
          "weigher can not be combined with maximum size (%s provided)",
          this.maximumSize);
    }

    // safely limiting the kinds of caches this can produce
    @SuppressWarnings("unchecked")
