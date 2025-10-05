// Source-based slice around line 283
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testHashSet()>

  public void testLinkedList() {
    assertFreshInstance(new TypeToken<LinkedList<String>>() {});
  }

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
