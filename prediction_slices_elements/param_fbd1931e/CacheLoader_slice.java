// Source-based slice around line 140
// Method: <com.google.common.cache.CacheLoader: CacheLoader from(Function)>

  /**
   * Returns a cache loader that uses {@code function} to load keys, without supporting either
   * reloading or bulk loading. This allows creating a cache loader using a lambda expression.
   *
   * <p>The returned object is serializable if {@code function} is serializable.
   *
   * @param function the function to be used for loading values; must never return {@code null}
   * @return a cache loader that loads values by passing each key to {@code function}
   */
  public static <K, V> CacheLoader<K, V> from(Function<K, V> function) {
    return new FunctionToCacheLoader<>(function);
  }

  /**
   * Returns a cache loader based on an <i>existing</i> supplier instance. Note that there's no need
   * to create a <i>new</i> supplier just to pass it in here; just subclass {@code CacheLoader} and
   * implement {@link #load load} instead.
   *
   * <p>The returned object is serializable if {@code supplier} is serializable.
   *
