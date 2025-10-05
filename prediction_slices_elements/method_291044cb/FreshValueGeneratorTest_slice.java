// Source-based slice around line 248
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableListMultimap()>

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
  }

  public void testImmutableBiMap() {
    assertFreshInstance(new TypeToken<ImmutableBiMap<String, Integer>>() {});
  }
