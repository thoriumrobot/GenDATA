// Source-based slice around line 269
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testArrayList()>

  public void testImmutableTable() {
    assertFreshInstance(new TypeToken<ImmutableTable<String, Integer, ImmutableList<String>>>() {});
  }

  public void testList() {
    assertFreshInstance(new TypeToken<List<String>>() {});
    assertNotInstantiable(new TypeToken<List<EmptyEnum>>() {});
  }

  public void testArrayList() {
    assertFreshInstance(new TypeToken<ArrayList<String>>() {});
    assertNotInstantiable(new TypeToken<ArrayList<EmptyEnum>>() {});
  }

  public void testLinkedList() {
    assertFreshInstance(new TypeToken<LinkedList<String>>() {});
  }

  public void testSet() {
    assertFreshInstance(new TypeToken<Set<String>>() {});
