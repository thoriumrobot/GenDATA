// Source-based slice around line 299
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testNavigableSet()>


  public void testTreeSet() {
    assertFreshInstance(new TypeToken<TreeSet<String>>() {});
  }

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
