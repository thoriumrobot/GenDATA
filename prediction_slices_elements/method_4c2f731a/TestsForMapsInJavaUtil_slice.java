// Source-based slice around line 401
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForConcurrentHashMap()>

            MapFeature.RESTRICTS_KEYS,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.KNOWN_ORDER,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForEnumMap())
        .createTestSuite();
  }

  public Test testsForConcurrentHashMap() {
    return ConcurrentMapTestSuiteBuilder.using(
            new TestStringMapGenerator() {
              @Override
              protected Map<String, String> create(Entry<String, String>[] entries) {
                return populate(new ConcurrentHashMap<String, String>(), entries);
              }
            })
        .named("ConcurrentHashMap")
        .withFeatures(
            MapFeature.GENERAL_PURPOSE,
