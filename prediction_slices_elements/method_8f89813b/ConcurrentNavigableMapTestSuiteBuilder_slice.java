// Source-based slice around line 51
// Method: <com.google.common.collect.testing.ConcurrentNavigableMapTestSuiteBuilder: NavigableMapTestSuiteBuilder subSuiteUsing(TestSortedMapGenerator)>

  @SuppressWarnings("rawtypes") // class literals
  @Override
  protected List<Class<? extends AbstractTester>> getTesters() {
    List<Class<? extends AbstractTester>> testers = copyToList(super.getTesters());
    testers.addAll(ConcurrentMapTestSuiteBuilder.TESTERS);
    return testers;
  }

  @Override
  NavigableMapTestSuiteBuilder<K, V> subSuiteUsing(TestSortedMapGenerator<K, V> generator) {
    return using(generator);
  }
}
