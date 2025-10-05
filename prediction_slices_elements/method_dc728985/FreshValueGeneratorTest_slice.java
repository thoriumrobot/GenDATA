// Source-based slice around line 230
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableCollection()>

  public void testImmutableSortedSet() {
    assertFreshInstance(new TypeToken<ImmutableSortedSet<String>>() {});
  }

  public void testImmutableMultiset() {
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
