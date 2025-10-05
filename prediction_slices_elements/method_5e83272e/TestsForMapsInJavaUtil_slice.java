// Source-based slice around line 293
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForTreeMapNatural()>

            MapFeature.FAILS_FAST_ON_CONCURRENT_MODIFICATION,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.KNOWN_ORDER,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForLinkedHashMap())
        .createTestSuite();
  }

  public Test testsForTreeMapNatural() {
    return NavigableMapTestSuiteBuilder.using(
            new TestStringSortedMapGenerator() {
              @Override
              protected SortedMap<String, String> create(Entry<String, String>[] entries) {
                /*
                 * TODO(cpovirk): it would be nice to create an input Map and use
                 * the copy constructor here and in the other tests
                 */
                return populate(new TreeMap<String, String>(), entries);
              }
