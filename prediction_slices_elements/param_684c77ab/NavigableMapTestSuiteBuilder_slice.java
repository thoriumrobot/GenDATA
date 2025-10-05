// Source-based slice around line 83
// Method: <com.google.common.collect.testing.NavigableMapTestSuiteBuilder: NavigableSetTestSuiteBuilder createDerivedKeySetSuite(TestSetGenerator)>

      derivedSuites.add(createSubmapSuite(parentBuilder, Bound.EXCLUSIVE, Bound.INCLUSIVE));
      derivedSuites.add(createSubmapSuite(parentBuilder, Bound.INCLUSIVE, Bound.INCLUSIVE));
    }

    return derivedSuites;
  }

  @Override
  protected NavigableSetTestSuiteBuilder<K> createDerivedKeySetSuite(
      TestSetGenerator<K> keySetGenerator) {
    return NavigableSetTestSuiteBuilder.using((TestSortedSetGenerator<K>) keySetGenerator);
  }

  public static final class NavigableMapSubmapTestMapGenerator<K, V>
      extends SortedMapSubmapTestMapGenerator<K, V> {
    public NavigableMapSubmapTestMapGenerator(
        TestSortedMapGenerator<K, V> delegate, Bound to, Bound from) {
      super(delegate, to, from);
    }

