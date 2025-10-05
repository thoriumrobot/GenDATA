// Source-based slice around line 359
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testConcurrentMap()>


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
    assertFreshInstance(new TypeToken<Multimap<String, ?>>() {});
  }

  public void testHashMultimap() {
