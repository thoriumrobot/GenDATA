// Source-based slice around line 41
// Method: <com.google.common.collect.testing.ReserializedSafeTreeMapMapInterfaceTest: SortedMap makeEmptyMap()>

  protected SortedMap<String, Integer> makePopulatedMap() {
    NavigableMap<String, Integer> map = new SafeTreeMap<>();
    map.put("one", 1);
    map.put("two", 2);
    map.put("three", 3);
    return SerializableTester.reserialize(map);
  }

  @Override
  protected SortedMap<String, Integer> makeEmptyMap() throws UnsupportedOperationException {
    NavigableMap<String, Integer> map = new SafeTreeMap<>();
    return SerializableTester.reserialize(map);
  }

  @Override
  protected String getKeyNotInPopulatedMap() {
    return "minus one";
  }

  @Override
