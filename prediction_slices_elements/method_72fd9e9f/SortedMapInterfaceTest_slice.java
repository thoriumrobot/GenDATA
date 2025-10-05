// Source-based slice around line 52
// Method: <com.google.common.collect.testing.SortedMapInterfaceTest: SortedMap makeEitherMap()>

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

  public void testTailMapWriteThrough() {
    SortedMap<K, V> map;
    try {
