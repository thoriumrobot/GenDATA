// Source-based slice around line 223
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForHashMap()>

            MapFeature.ALLOWS_NULL_KEYS,
            MapFeature.ALLOWS_NULL_VALUES,
            MapFeature.ALLOWS_ANY_NULL_QUERIES,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ONE)
        .suppressing(suppressForSingletonMap())
        .createTestSuite();
  }

  public Test testsForHashMap() {
    return MapTestSuiteBuilder.using(
            new TestStringMapGenerator() {
              @Override
              protected Map<String, String> create(Entry<String, String>[] entries) {
                return toHashMap(entries);
              }
            })
        .named("HashMap")
        .withFeatures(
            MapFeature.GENERAL_PURPOSE,
