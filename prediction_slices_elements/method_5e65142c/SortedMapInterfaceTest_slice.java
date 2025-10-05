// Source-based slice around line 105
// Method: <com.google.common.collect.testing.SortedMapInterfaceTest: void testTailMapClearThrough()>

    K key = secondEntry.getKey();
    SortedMap<K, V> subMap = map.tailMap(key);
    subMap.remove(key);
    assertNull(subMap.remove(firstEntry.getKey()));
    assertEquals(map.size(), oldSize - 1);
    assertFalse(map.containsKey(key));
    assertEquals(subMap.size(), oldSize - 2);
  }

  public void testTailMapClearThrough() {
    SortedMap<K, V> map;
    try {
      map = makePopulatedMap();
    } catch (UnsupportedOperationException e) {
      return;
    }
    int oldSize = map.size();
    if (map.size() < 2 || !supportsClear) {
      return;
    }
