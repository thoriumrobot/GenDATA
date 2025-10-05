// Source-based slice around line 155
// Method: <com.google.common.cache.CacheLoader: CacheLoader from(Supplier)>

   * to create a <i>new</i> supplier just to pass it in here; just subclass {@code CacheLoader} and
   * implement {@link #load load} instead.
   *
   * <p>The returned object is serializable if {@code supplier} is serializable.
   *
   * @param supplier the supplier to be used for loading values; must never return {@code null}
   * @return a cache loader that loads values by calling {@link Supplier#get}, irrespective of the
   *     key
   */
  public static <V> CacheLoader<Object, V> from(Supplier<V> supplier) {
    return new SupplierToCacheLoader<>(supplier);
  }

  private static final class FunctionToCacheLoader<K, V> extends CacheLoader<K, V>
      implements Serializable {
    private final Function<K, V> computingFunction;

    FunctionToCacheLoader(Function<K, V> computingFunction) {
      this.computingFunction = checkNotNull(computingFunction);
    }
