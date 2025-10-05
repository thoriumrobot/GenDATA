// Source-based slice around line 44
// Method: <com.google.common.collect.testing.ConcurrentNavigableMapTestSuiteBuilder: List getTesters()>

      TestSortedMapGenerator<K, V> generator) {
    ConcurrentNavigableMapTestSuiteBuilder<K, V> result =
        new ConcurrentNavigableMapTestSuiteBuilder<>();
    result.usingGenerator(generator);
    return result;
  }

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
