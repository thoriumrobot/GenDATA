// Source-based slice around line 380
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForEnumMap()>

        .withFeatures(
            MapFeature.ALLOWS_NULL_VALUES,
            CollectionFeature.KNOWN_ORDER,
            CollectionFeature.SERIALIZABLE,
            CollectionSize.ANY)
        .suppressing(suppressForUnmodifiableSortedMap())
        .createTestSuite();
  }

  public Test testsForEnumMap() {
    return MapTestSuiteBuilder.using(
            new TestEnumMapGenerator() {
              @Override
              protected Map<AnEnum, String> create(Entry<AnEnum, String>[] entries) {
                return populate(new EnumMap<AnEnum, String>(AnEnum.class), entries);
              }
            })
        .named("EnumMap")
        .withFeatures(
            MapFeature.GENERAL_PURPOSE,
