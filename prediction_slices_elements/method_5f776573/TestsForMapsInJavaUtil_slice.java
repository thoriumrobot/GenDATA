// Source-based slice around line 82
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Collection suppressForCheckedMap()>

    suite.addTest(testsForUnmodifiableMap());
    suite.addTest(testsForUnmodifiableSortedMap());
    suite.addTest(testsForEnumMap());
    suite.addTest(testsForConcurrentHashMap());
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
