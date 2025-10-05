// Source-based slice around line 270
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForLinkedHashMap()>

            CollectionFeature.SERIALIZABLE,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionFeature.SUPPORTS_REMOVE,
            CollectionSize.ANY)
        .named("Hashtable")
        .suppressing(suppressForHashtable())
        .createTestSuite();
  }

  public Test testsForLinkedHashMap() {
    return MapTestSuiteBuilder.using(
            new TestStringMapGenerator() {
              @Override
              protected Map<String, String> create(Entry<String, String>[] entries) {
                return populate(new LinkedHashMap<String, String>(), entries);
              }
            })
        .named("LinkedHashMap")
        .withFeatures(
            MapFeature.GENERAL_PURPOSE,
