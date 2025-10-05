// Source-based slice around line 355
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testNavigableMap()>


  public void testTreeMap() {
    assertFreshInstance(new TypeToken<TreeMap<String, ?>>() {});
  }

  public void testSortedMap() {
    assertFreshInstance(new TypeToken<SortedMap<?, String>>() {});
  }

  public void testNavigableMap() {
    assertFreshInstance(new TypeToken<NavigableMap<?, ?>>() {});
  }

  public void testConcurrentMap() {
    assertFreshInstance(new TypeToken<ConcurrentMap<String, ?>>() {});
    assertCanGenerateOnly(
        new TypeToken<ConcurrentMap<EmptyEnum, String>>() {}, Maps.newConcurrentMap());
  }

  public void testMultimap() {
