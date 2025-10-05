// Source-based slice around line 166
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForCheckedSortedMap()>

            MapFeature.RESTRICTS_KEYS,
            MapFeature.RESTRICTS_VALUES,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForCheckedMap())
        .createTestSuite();
  }

  public Test testsForCheckedSortedMap() {
    return SortedMapTestSuiteBuilder.using(
            new TestStringSortedMapGenerator() {
              @Override
              protected SortedMap<String, String> create(Entry<String, String>[] entries) {
                SortedMap<String, String> map = populate(new TreeMap<String, String>(), entries);
                return Collections.checkedSortedMap(map, String.class, String.class);
              }
            })
        .named("checkedSortedMap/TreeMap, natural")
        .withFeatures(
