// Source-based slice around line 264
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testList()>


  public void testImmutableBiMap() {
    assertFreshInstance(new TypeToken<ImmutableBiMap<String, Integer>>() {});
  }

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
