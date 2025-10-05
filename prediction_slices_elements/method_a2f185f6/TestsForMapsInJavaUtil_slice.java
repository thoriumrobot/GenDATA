// Source-based slice around line 130
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Collection suppressForConcurrentHashMap()>


  protected Collection<Method> suppressForUnmodifiableSortedMap() {
    return emptySet();
  }

  protected Collection<Method> suppressForEnumMap() {
    return emptySet();
  }

  protected Collection<Method> suppressForConcurrentHashMap() {
    return emptySet();
  }

  protected Collection<Method> suppressForConcurrentSkipListMap() {
    return asList(
        MapEntrySetTester.getSetValueMethod(),
        MapEntrySetTester.getSetValueWithNullValuesAbsentMethod(),
        MapEntrySetTester.getSetValueWithNullValuesPresentMethod());
  }

