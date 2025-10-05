// Source-based slice around line 126
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Collection suppressForEnumMap()>


  protected Collection<Method> suppressForUnmodifiableMap() {
    return emptySet();
  }

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
