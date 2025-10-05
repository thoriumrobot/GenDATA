// Source-based slice around line 373
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testLinkedHashMultimap()>


  public void testMultimap() {
    assertFreshInstance(new TypeToken<Multimap<String, ?>>() {});
  }

  public void testHashMultimap() {
    assertFreshInstance(new TypeToken<HashMultimap<String, ?>>() {});
  }

  public void testLinkedHashMultimap() {
    assertFreshInstance(new TypeToken<LinkedHashMultimap<String, ?>>() {});
  }

  public void testListMultimap() {
    assertFreshInstance(new TypeToken<ListMultimap<String, ?>>() {});
  }

  public void testArrayListMultimap() {
    assertFreshInstance(new TypeToken<ArrayListMultimap<String, ?>>() {});
  }
