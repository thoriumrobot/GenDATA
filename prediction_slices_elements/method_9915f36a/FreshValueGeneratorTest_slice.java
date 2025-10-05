// Source-based slice around line 243
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableMultimap()>


  public void testImmutableMap() {
    assertFreshInstance(new TypeToken<ImmutableMap<String, Integer>>() {});
  }

  public void testImmutableSortedMap() {
    assertFreshInstance(new TypeToken<ImmutableSortedMap<String, Integer>>() {});
  }

  public void testImmutableMultimap() {
    assertFreshInstance(new TypeToken<ImmutableMultimap<String, Integer>>() {});
    assertNotInstantiable(new TypeToken<ImmutableMultimap<EmptyEnum, String>>() {});
  }

  public void testImmutableListMultimap() {
    assertFreshInstance(new TypeToken<ImmutableListMultimap<String, Integer>>() {});
  }

  public void testImmutableSetMultimap() {
    assertFreshInstance(new TypeToken<ImmutableSetMultimap<String, Integer>>() {});
