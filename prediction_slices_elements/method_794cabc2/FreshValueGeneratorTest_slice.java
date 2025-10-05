// Source-based slice around line 235
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableMap()>

    assertFreshInstance(new TypeToken<ImmutableSortedSet<String>>() {});
    assertNotInstantiable(new TypeToken<ImmutableMultiset<EmptyEnum>>() {});
  }

  public void testImmutableCollection() {
    assertFreshInstance(new TypeToken<ImmutableCollection<String>>() {});
    assertNotInstantiable(new TypeToken<ImmutableCollection<EmptyEnum>>() {});
  }

  public void testImmutableMap() {
    assertFreshInstance(new TypeToken<ImmutableMap<String, Integer>>() {});
  }

  public void testImmutableSortedMap() {
    assertFreshInstance(new TypeToken<ImmutableSortedMap<String, Integer>>() {});
  }

  public void testImmutableMultimap() {
    assertFreshInstance(new TypeToken<ImmutableMultimap<String, Integer>>() {});
    assertNotInstantiable(new TypeToken<ImmutableMultimap<EmptyEnum, String>>() {});
