// Source-based slice around line 353
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder lenientParsing()>

  }

  /**
   * Enables lenient parsing. Useful for tests and spec parsing.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
   */
  @GwtIncompatible // To be supported
  @CanIgnoreReturnValue
  CacheBuilder<K, V> lenientParsing() {
    strictParsing = false;
    return this;
  }

  /**
   * Sets a custom {@code Equivalence} strategy for comparing keys.
   *
   * <p>By default, the cache uses {@link Equivalence#identity} to determine key equality when
   * {@link #weakKeys} is specified, and {@link Equivalence#equals()} otherwise.
   *
