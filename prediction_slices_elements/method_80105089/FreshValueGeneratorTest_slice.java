// Source-based slice around line 319
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testTreeMultiset()>


  public void testHashMultiset() {
    assertFreshInstance(new TypeToken<HashMultiset<String>>() {});
  }

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
