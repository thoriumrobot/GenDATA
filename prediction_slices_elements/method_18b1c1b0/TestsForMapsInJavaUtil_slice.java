// Source-based slice around line 204
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test testsForSingletonMap()>

                return emptyMap();
              }
            })
        .named("emptyMap")
        .withFeatures(CollectionFeature.SERIALIZABLE, CollectionSize.ZERO)
        .suppressing(suppressForEmptyMap())
        .createTestSuite();
  }

  public Test testsForSingletonMap() {
    return MapTestSuiteBuilder.using(
            new TestStringMapGenerator() {
              @Override
              protected Map<String, String> create(Entry<String, String>[] entries) {
                return singletonMap(entries[0].getKey(), entries[0].getValue());
              }
            })
        .named("singletonMap")
        .withFeatures(
            MapFeature.ALLOWS_NULL_KEYS,
