// Source-based slice around line 327
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testCollection()>


  public void testTreeMultiset() {
    assertFreshInstance(new TypeToken<TreeMultiset<String>>() {});
  }

  public void testImmutableSortedMultiset() {
    assertFreshInstance(new TypeToken<ImmutableSortedMultiset<String>>() {});
  }

  public void testCollection() {
    assertFreshInstance(new TypeToken<Collection<String>>() {});
  }

  public void testIterable() {
    assertFreshInstance(new TypeToken<Iterable<String>>() {});
  }

  public void testMap() {
    assertFreshInstance(new TypeToken<Map<String, ?>>() {});
  }
