// Source-based slice around line 141
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForCheckedMap()>

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
              protected Map<String, String> create(Entry<String, String>[] entries) {
                Map<String, String> map = populate(new HashMap<String, String>(), entries);
                return Collections.checkedMap(map, String.class, String.class);
              }
            })
        .named("checkedMap/HashMap")
        .withFeatures(
