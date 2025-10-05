// Source-based slice around line 114
// Method: <com.google.common.collect.testing.NavigableMapTestSuiteBuilder: NavigableMapTestSuiteBuilder newBuilderUsing(TestSortedMapGenerator,Bound,Bound)>

      } else if (from == Bound.INCLUSIVE && to == Bound.INCLUSIVE) {
        return map.subMap(firstInclusive, true, lastInclusive, true);
      } else {
        return (NavigableMap<K, V>) super.createSubMap(map, firstExclusive, lastExclusive);
      }
    }
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
