// Source-based slice around line 52
// Method: <com.google.common.collect.testing.NavigableMapTestSuiteBuilder: List getTesters()>

  public static <K, V> NavigableMapTestSuiteBuilder<K, V> using(
      TestSortedMapGenerator<K, V> generator) {
    NavigableMapTestSuiteBuilder<K, V> result = new NavigableMapTestSuiteBuilder<>();
    result.usingGenerator(generator);
    return result;
  }

  @SuppressWarnings("rawtypes") // class literals
  @Override
  protected List<Class<? extends AbstractTester>> getTesters() {
    List<Class<? extends AbstractTester>> testers = copyToList(super.getTesters());
    testers.add(NavigableMapNavigationTester.class);
    return testers;
  }

  @Override
  protected List<TestSuite> createDerivedSuites(
      FeatureSpecificTestSuiteBuilder<
              ?, ? extends OneSizeTestContainerGenerator<Map<K, V>, Entry<K, V>>>
          parentBuilder) {
