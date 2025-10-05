// Source-based slice around line 342
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForUnmodifiableMap()>

            MapFeature.FAILS_FAST_ON_CONCURRENT_MODIFICATION,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.KNOWN_ORDER,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForTreeMapWithComparator())
        .createTestSuite();
  }

  public Test testsForUnmodifiableMap() {
    return MapTestSuiteBuilder.using(
            new TestStringMapGenerator() {
              @Override
              protected Map<String, String> create(Entry<String, String>[] entries) {
                return unmodifiableMap(toHashMap(entries));
              }
            })
        .named("unmodifiableMap/HashMap")
        .withFeatures(
            MapFeature.ALLOWS_NULL_KEYS,
