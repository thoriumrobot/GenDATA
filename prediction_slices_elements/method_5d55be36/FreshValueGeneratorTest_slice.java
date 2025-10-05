// Source-based slice around line 278
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testSet()>

  public void testArrayList() {
    assertFreshInstance(new TypeToken<ArrayList<String>>() {});
    assertNotInstantiable(new TypeToken<ArrayList<EmptyEnum>>() {});
  }

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
