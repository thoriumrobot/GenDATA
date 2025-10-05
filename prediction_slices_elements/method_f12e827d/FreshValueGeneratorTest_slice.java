// Source-based slice around line 221
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableSortedSet()>


  public void testImmutableList() {
    assertFreshInstance(new TypeToken<ImmutableList<String>>() {});
  }

  public void testImmutableSet() {
    assertFreshInstance(new TypeToken<ImmutableSet<String>>() {});
  }

  public void testImmutableSortedSet() {
    assertFreshInstance(new TypeToken<ImmutableSortedSet<String>>() {});
  }

  public void testImmutableMultiset() {
    assertFreshInstance(new TypeToken<ImmutableSortedSet<String>>() {});
    assertNotInstantiable(new TypeToken<ImmutableMultiset<EmptyEnum>>() {});
  }

  public void testImmutableCollection() {
    assertFreshInstance(new TypeToken<ImmutableCollection<String>>() {});
