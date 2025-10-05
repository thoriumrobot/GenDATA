// Source-based slice around line 385
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testSetMultimap()>


  public void testListMultimap() {
    assertFreshInstance(new TypeToken<ListMultimap<String, ?>>() {});
  }

  public void testArrayListMultimap() {
    assertFreshInstance(new TypeToken<ArrayListMultimap<String, ?>>() {});
  }

  public void testSetMultimap() {
    assertFreshInstance(new TypeToken<SetMultimap<String, ?>>() {});
  }

  public void testBiMap() {
    assertFreshInstance(new TypeToken<BiMap<String, ?>>() {});
    assertNotInstantiable(new TypeToken<BiMap<EmptyEnum, String>>() {});
  }

  public void testHashBiMap() {
    assertFreshInstance(new TypeToken<HashBiMap<String, ?>>() {});
