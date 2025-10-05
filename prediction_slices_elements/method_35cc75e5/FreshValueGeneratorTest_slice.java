// Source-based slice around line 339
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testHashMap()>


  public void testIterable() {
    assertFreshInstance(new TypeToken<Iterable<String>>() {});
  }

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
