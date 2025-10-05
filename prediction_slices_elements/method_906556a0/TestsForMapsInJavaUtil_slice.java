// Source-based slice around line 134
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Collection suppressForConcurrentSkipListMap()>


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

  public Test testsForCheckedMap() {
    return MapTestSuiteBuilder.using(
            new TestStringMapGenerator() {
              @Override
