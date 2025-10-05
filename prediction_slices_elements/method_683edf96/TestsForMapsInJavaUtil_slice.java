// Source-based slice around line 419
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForConcurrentSkipListMapNatural()>

        .withFeatures(
            MapFeature.GENERAL_PURPOSE,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForConcurrentHashMap())
        .createTestSuite();
  }

  public Test testsForConcurrentSkipListMapNatural() {
    return ConcurrentNavigableMapTestSuiteBuilder.using(
            new TestStringSortedMapGenerator() {
              @Override
              protected SortedMap<String, String> create(Entry<String, String>[] entries) {
                return populate(new ConcurrentSkipListMap<String, String>(), entries);
              }
            })
        .named("ConcurrentSkipListMap, natural")
        .withFeatures(
            MapFeature.GENERAL_PURPOSE,
