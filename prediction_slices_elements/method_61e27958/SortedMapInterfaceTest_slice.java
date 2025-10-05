// Source-based slice around line 46
// Method: <com.google.common.collect.testing.SortedMapInterfaceTest: SortedMap makeEmptyMap()>

      boolean allowsNullKeys,
      boolean allowsNullValues,
      boolean supportsPut,
      boolean supportsRemove,
      boolean supportsClear) {
    super(allowsNullKeys, allowsNullValues, supportsPut, supportsRemove, supportsClear);
  }

  @Override
  protected abstract SortedMap<K, V> makeEmptyMap() throws UnsupportedOperationException;

  @Override
  protected abstract SortedMap<K, V> makePopulatedMap() throws UnsupportedOperationException;

  @Override
  protected SortedMap<K, V> makeEitherMap() {
    try {
      return makePopulatedMap();
    } catch (UnsupportedOperationException e) {
      return makeEmptyMap();
