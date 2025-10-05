// Source-based slice around line 411
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testTreeBasedTable()>


  public void testHashBasedTable() {
    assertFreshInstance(new TypeToken<HashBasedTable<String, ?, ?>>() {});
  }

  public void testRowSortedTable() {
    assertFreshInstance(new TypeToken<RowSortedTable<String, ?, ?>>() {});
  }

  public void testTreeBasedTable() {
    assertFreshInstance(new TypeToken<TreeBasedTable<String, ?, ?>>() {});
  }

  public void testObject() {
    assertEquals(
        new FreshValueGenerator().generateFresh(String.class),
        new FreshValueGenerator().generateFresh(Object.class));
  }

  public void testEnums() {
