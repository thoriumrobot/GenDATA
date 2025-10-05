// Source-based slice around line 44
// Method: <com.google.common.collect.testing.MinimalIterableTest: void testOf_one()>


  public void testOf_empty() {
    Iterable<String> iterable = MinimalIterable.<String>of();
    Iterator<String> iterator = iterable.iterator();
    assertFalse(iterator.hasNext());
    assertThrows(NoSuchElementException.class, () -> iterator.next());
    assertThrows(IllegalStateException.class, () -> iterable.iterator());
  }

  public void testOf_one() {
    Iterable<String> iterable = MinimalIterable.of("a");
    Iterator<String> iterator = iterable.iterator();
    assertTrue(iterator.hasNext());
    assertEquals("a", iterator.next());
    assertFalse(iterator.hasNext());
    assertThrows(NoSuchElementException.class, () -> iterator.next());
    assertThrows(IllegalStateException.class, () -> iterable.iterator());
  }

  public void testFrom_empty() {
