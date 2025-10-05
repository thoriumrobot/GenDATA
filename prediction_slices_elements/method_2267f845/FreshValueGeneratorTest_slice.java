// Source-based slice around line 381
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testArrayListMultimap()>


  public void testLinkedHashMultimap() {
    assertFreshInstance(new TypeToken<LinkedHashMultimap<String, ?>>() {});
  }

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
