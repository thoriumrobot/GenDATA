// Source-based slice around line 190
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForEmptyMap()>

            MapFeature.RESTRICTS_VALUES,
            CollectionFeature.KNOWN_ORDER,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForCheckedSortedMap())
        .createTestSuite();
  }

  public Test testsForEmptyMap() {
    return MapTestSuiteBuilder.using(
            new TestStringMapGenerator() {
              @Override
              protected Map<String, String> create(Entry<String, String>[] entries) {
                return emptyMap();
              }
            })
        .named("emptyMap")
        .withFeatures(CollectionFeature.SERIALIZABLE, CollectionSize.ZERO)
        .suppressing(suppressForEmptyMap())
