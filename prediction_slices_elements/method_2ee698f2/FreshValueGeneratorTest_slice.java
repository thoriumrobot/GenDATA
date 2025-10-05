// Source-based slice around line 252
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableSetMultimap()>

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

  public void testImmutableTable() {
    assertFreshInstance(new TypeToken<ImmutableTable<String, Integer, ImmutableList<String>>>() {});
  }
