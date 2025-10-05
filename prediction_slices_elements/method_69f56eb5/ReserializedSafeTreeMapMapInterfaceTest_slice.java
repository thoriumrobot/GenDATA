// Source-based slice around line 47
// Method: <com.google.common.collect.testing.ReserializedSafeTreeMapMapInterfaceTest: String getKeyNotInPopulatedMap()>

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
  protected Integer getValueNotInPopulatedMap() {
    return -1;
  }
}
