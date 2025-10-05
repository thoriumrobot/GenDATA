// Source-based slice around line 49
// Method: <com.google.common.collect.testing.SortedMapInterfaceTest: SortedMap makePopulatedMap()>

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
    }
  }

