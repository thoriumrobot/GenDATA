// Source-based slice around line 307
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testSortedMultiset()>


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

  public void testLinkedHashMultiset() {
    assertFreshInstance(new TypeToken<LinkedHashMultiset<String>>() {});
  }
