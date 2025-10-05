// Source-based slice around line 260
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableTable()>


  public void testImmutableSetMultimap() {
    assertFreshInstance(new TypeToken<ImmutableSetMultimap<String, Integer>>() {});
  }

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
