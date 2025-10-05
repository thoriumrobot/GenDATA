// Source-based slice around line 303
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testMultiset()>


  public void testSortedSet() {
    assertFreshInstance(new TypeToken<SortedSet<String>>() {});
  }

  public void testNavigableSet() {
    assertFreshInstance(new TypeToken<NavigableSet<String>>() {});
  }

  public void testMultiset() {
    assertFreshInstance(new TypeToken<Multiset<String>>() {});
  }

  public void testSortedMultiset() {
    assertFreshInstance(new TypeToken<SortedMultiset<String>>() {});
  }

  public void testHashMultiset() {
    assertFreshInstance(new TypeToken<HashMultiset<String>>() {});
  }
