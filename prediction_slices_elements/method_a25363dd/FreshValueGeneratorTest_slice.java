// Source-based slice around line 291
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testTreeSet()>


  public void testHashSet() {
    assertFreshInstance(new TypeToken<HashSet<String>>() {});
  }

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
