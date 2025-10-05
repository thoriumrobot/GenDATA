// Source-based slice around line 311
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testHashMultiset()>


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

  public void testTreeMultiset() {
    assertFreshInstance(new TypeToken<TreeMultiset<String>>() {});
  }
