// Source-based slice around line 361
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForUnmodifiableSortedMap()>

            MapFeature.ALLOWS_NULL_KEYS,
            MapFeature.ALLOWS_NULL_VALUES,
            MapFeature.ALLOWS_ANY_NULL_QUERIES,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForUnmodifiableMap())
        .createTestSuite();
  }

  public Test testsForUnmodifiableSortedMap() {
    return MapTestSuiteBuilder.using(
            new TestStringSortedMapGenerator() {
              @Override
              protected SortedMap<String, String> create(Entry<String, String>[] entries) {
                SortedMap<String, String> map = populate(new TreeMap<String, String>(), entries);
                return Collections.unmodifiableSortedMap(map);
              }
            })
        .named("unmodifiableSortedMap/TreeMap, natural")
        .withFeatures(
