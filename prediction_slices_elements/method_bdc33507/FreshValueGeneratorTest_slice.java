// Source-based slice around line 287
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testLinkedHashSet()>

  public void testSet() {
    assertFreshInstance(new TypeToken<Set<String>>() {});
    assertNotInstantiable(new TypeToken<Set<EmptyEnum>>() {});
  }

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
