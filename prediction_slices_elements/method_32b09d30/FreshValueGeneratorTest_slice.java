// Source-based slice around line 351
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testSortedMap()>


  public void testLinkedHashMap() {
    assertFreshInstance(new TypeToken<LinkedHashMap<String, ?>>() {});
  }

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
