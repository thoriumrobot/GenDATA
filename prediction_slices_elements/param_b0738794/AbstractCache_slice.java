// Source-based slice around line 68
// Method: <com.google.common.cache.AbstractCache: ImmutableMap getAllPresent(Iterable)>

   * possible with an unsafe cast which requires {@code keys} to actually be of type {@code K}.
   *
   * @since 11.0
   */
  /*
   * <? extends Object> is mostly the same as <?> to plain Java. But to nullness checkers, they
   * differ: <? extends Object> means "non-null types," while <?> means "all types."
   */
  @Override
  public ImmutableMap<K, V> getAllPresent(Iterable<? extends Object> keys) {
    Map<K, V> result = new LinkedHashMap<>();
    for (Object key : keys) {
      if (!result.containsKey(key)) {
        @SuppressWarnings("unchecked")
        K castKey = (K) key;
        V value = getIfPresent(key);
        if (value != null) {
          result.put(castKey, value);
        }
      }
