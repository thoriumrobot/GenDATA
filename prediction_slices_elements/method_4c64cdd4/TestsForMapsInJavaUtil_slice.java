// Source-based slice around line 318
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForTreeMapWithComparator()>

            MapFeature.FAILS_FAST_ON_CONCURRENT_MODIFICATION,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.KNOWN_ORDER,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForTreeMapNatural())
        .createTestSuite();
  }

  public Test testsForTreeMapWithComparator() {
    return NavigableMapTestSuiteBuilder.using(
            new TestStringSortedMapGenerator() {
              @Override
              protected SortedMap<String, String> create(Entry<String, String>[] entries) {
                return populate(
                    new TreeMap<String, String>(arbitraryNullFriendlyComparator()), entries);
              }
            })
        .named("TreeMap, with comparator")
        .withFeatures(
