// Source-based slice around line 245
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForHashtable()>

            MapFeature.ALLOWS_ANY_NULL_QUERIES,
            MapFeature.FAILS_FAST_ON_CONCURRENT_MODIFICATION,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForHashMap())
        .createTestSuite();
  }

  public Test testsForHashtable() {
    return MapTestSuiteBuilder.using(
            new TestStringMapGenerator() {
              @Override
              // We are testing Hashtable / testing our tests on Hashtable.
              @SuppressWarnings("JdkObsolete")
              protected Map<String, String> create(Entry<String, String>[] entries) {
                return populate(new Hashtable<String, String>(), entries);
              }
            })
        .withFeatures(
