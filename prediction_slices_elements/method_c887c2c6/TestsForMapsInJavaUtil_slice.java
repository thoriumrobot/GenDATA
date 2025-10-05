// Source-based slice around line 438
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForConcurrentSkipListMapWithComparator()>

            MapFeature.GENERAL_PURPOSE,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.KNOWN_ORDER,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForConcurrentSkipListMap())
        .createTestSuite();
  }

  public Test testsForConcurrentSkipListMapWithComparator() {
    return ConcurrentNavigableMapTestSuiteBuilder.using(
            new TestStringSortedMapGenerator() {
              @Override
              protected SortedMap<String, String> create(Entry<String, String>[] entries) {
                return populate(
                    new ConcurrentSkipListMap<String, String>(arbitraryNullFriendlyComparator()),
                    entries);
              }
            })
        .named("ConcurrentSkipListMap, with comparator")
