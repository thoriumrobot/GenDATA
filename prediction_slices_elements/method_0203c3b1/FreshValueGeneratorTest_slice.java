// Source-based slice around line 407
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testRowSortedTable()>

  public void testTable() {
    assertFreshInstance(new TypeToken<Table<String, ?, ?>>() {});
    assertNotInstantiable(new TypeToken<Table<EmptyEnum, String, Integer>>() {});
  }

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
