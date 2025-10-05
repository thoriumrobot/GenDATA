// Source-based slice around line 323
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testImmutableSortedMultiset()>


  public void testLinkedHashMultiset() {
    assertFreshInstance(new TypeToken<LinkedHashMultiset<String>>() {});
  }

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
