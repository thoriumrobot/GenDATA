// Source-based slice around line 343
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testLinkedHashMap()>


  public void testMap() {
    assertFreshInstance(new TypeToken<Map<String, ?>>() {});
  }

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
