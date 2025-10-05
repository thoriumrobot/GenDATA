// Source-based slice around line 82
// Method: <com.google.common.collect.testing.SortedMapInterfaceTest: void testTailMapRemoveThrough()>

    K key = secondEntry.getKey();
    SortedMap<K, V> subMap = map.tailMap(key);
    V value = getValueNotInPopulatedMap();
    subMap.put(key, value);
    assertEquals(secondEntry.getValue(), value);
    assertEquals(map.get(key), value);
    assertThrows(IllegalArgumentException.class, () -> subMap.put(firstEntry.getKey(), value));
  }

  public void testTailMapRemoveThrough() {
    SortedMap<K, V> map;
    try {
      map = makePopulatedMap();
    } catch (UnsupportedOperationException e) {
      return;
    }
    int oldSize = map.size();
    if (map.size() < 2 || !supportsRemove) {
      return;
    }
