// Source-based slice around line 295
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testSortedSet()>


  public void testLinkedHashSet() {
    assertFreshInstance(new TypeToken<LinkedHashSet<String>>() {});
  }

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
