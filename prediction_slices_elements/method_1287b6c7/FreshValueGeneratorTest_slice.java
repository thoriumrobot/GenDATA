// Source-based slice around line 335
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testMap()>


  public void testCollection() {
    assertFreshInstance(new TypeToken<Collection<String>>() {});
  }

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
