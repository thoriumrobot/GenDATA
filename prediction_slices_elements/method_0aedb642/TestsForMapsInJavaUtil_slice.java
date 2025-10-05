// Source-based slice around line 86
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Collection suppressForCheckedSortedMap()>

    suite.addTest(testsForConcurrentSkipListMapNatural());
    suite.addTest(testsForConcurrentSkipListMapWithComparator());
    return suite;
  }

  protected Collection<Method> suppressForCheckedMap() {
    return emptySet();
  }

  protected Collection<Method> suppressForCheckedSortedMap() {
    return emptySet();
  }

  protected Collection<Method> suppressForEmptyMap() {
    return emptySet();
  }

  protected Collection<Method> suppressForSingletonMap() {
    return emptySet();
  }
