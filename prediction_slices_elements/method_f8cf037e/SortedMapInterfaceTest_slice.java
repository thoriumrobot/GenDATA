// Source-based slice around line 60
// Method: <com.google.common.collect.testing.SortedMapInterfaceTest: void testTailMapWriteThrough()>

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
      map = makePopulatedMap();
    } catch (UnsupportedOperationException e) {
      return;
    }
    if (map.size() < 2 || !supportsPut) {
      return;
    }
    Iterator<Entry<K, V>> iterator = map.entrySet().iterator();
