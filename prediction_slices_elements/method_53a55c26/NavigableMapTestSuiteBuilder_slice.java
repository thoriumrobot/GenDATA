// Source-based slice around line 138
// Method: <com.google.common.collect.testing.NavigableMapTestSuiteBuilder: NavigableMapTestSuiteBuilder subSuiteUsing(TestSortedMapGenerator)>

    features.addAll(parentBuilder.getFeatures());

    return subSuiteUsing(new DescendingTestMapGenerator<K, V>(delegate))
        .named(parentBuilder.getName() + " descending")
        .withFeatures(features)
        .suppressing(parentBuilder.getSuppressedTests())
        .createTestSuite();
  }

  NavigableMapTestSuiteBuilder<K, V> subSuiteUsing(TestSortedMapGenerator<K, V> generator) {
    return using(generator);
  }

  private static final class DescendingTestMapGenerator<K, V>
      extends ForwardingTestMapGenerator<K, V> implements TestSortedMapGenerator<K, V> {
    DescendingTestMapGenerator(TestSortedMapGenerator<K, V> delegate) {
      super(delegate);
    }

    @Override
