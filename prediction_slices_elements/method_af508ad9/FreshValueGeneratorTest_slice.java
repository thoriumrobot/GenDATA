// Source-based slice around line 347
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testTreeMap()>


  public void testHashMap() {
    assertFreshInstance(new TypeToken<HashMap<String, ?>>() {});
  }

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
