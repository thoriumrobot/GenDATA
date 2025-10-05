// Source-based slice around line 60
// Method: <com.google.common.collect.testing.NavigableMapTestSuiteBuilder: List createDerivedSuites(FeatureSpecificTestSuiteBuilder)>

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
    List<TestSuite> derivedSuites = super.createDerivedSuites(parentBuilder);

    if (!parentBuilder.getFeatures().contains(NoRecurse.DESCENDING)) {
      derivedSuites.add(createDescendingSuite(parentBuilder));
    }

    if (!parentBuilder.getFeatures().contains(NoRecurse.SUBMAP)) {
      // Other combinations are inherited from SortedMapTestSuiteBuilder.
