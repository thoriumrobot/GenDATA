// Source-based slice around line 256
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableBiMap()>


  public void testImmutableListMultimap() {
    assertFreshInstance(new TypeToken<ImmutableListMultimap<String, Integer>>() {});
  }

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
