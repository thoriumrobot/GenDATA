// Source-based slice around line 120
// Method: <com.google.common.collect.testing.NavigableMapTestSuiteBuilder: TestSuite createDescendingSuite(FeatureSpecificTestSuiteBuilder)>

  }

  @Override
  public NavigableMapTestSuiteBuilder<K, V> newBuilderUsing(
      TestSortedMapGenerator<K, V> delegate, Bound to, Bound from) {
    return subSuiteUsing(new NavigableMapSubmapTestMapGenerator<K, V>(delegate, to, from));
  }

  /** Create a suite whose maps are descending views of other maps. */
  private TestSuite createDescendingSuite(
      FeatureSpecificTestSuiteBuilder<
              ?, ? extends OneSizeTestContainerGenerator<Map<K, V>, Entry<K, V>>>
          parentBuilder) {
    TestSortedMapGenerator<K, V> delegate =
        (TestSortedMapGenerator<K, V>) parentBuilder.getSubjectGenerator().getInnerGenerator();

    List<Feature<?>> features = new ArrayList<>();
    features.add(NoRecurse.DESCENDING);
    features.addAll(parentBuilder.getFeatures());

