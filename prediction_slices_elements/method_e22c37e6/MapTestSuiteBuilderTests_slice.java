// Source-based slice around line 95
// Method: <com.google.common.collect.testing.MapTestSuiteBuilderTests: Test testsForHashMapNullKeysForbidden()>

        CollectionSize.ANY);
    return MapTestSuiteBuilder.using(generator)
        .named(name)
        .withFeatures(featuresList)
        .createTestSuite();
  }

  // TODO: consider being null-hostile in these tests

  private static Test testsForHashMapNullKeysForbidden() {
    return wrappedHashMapTests(
        new WrappedHashMapGenerator() {
          @Override
          Map<String, String> wrap(HashMap<String, String> map) {
            if (map.containsKey(null)) {
              throw new NullPointerException();
            }
            return new AbstractMap<String, String>() {
              @Override
              public Set<Entry<String, String>> entrySet() {
