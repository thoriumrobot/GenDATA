// Source-based slice around line 62
// Method: <com.google.common.collect.testing.MinimalIterableTest: void testFrom_one()>


  public void testFrom_empty() {
    Iterable<String> iterable = MinimalIterable.from(Collections.<String>emptySet());
    Iterator<String> iterator = iterable.iterator();
    assertFalse(iterator.hasNext());
    assertThrows(NoSuchElementException.class, () -> iterator.next());
    assertThrows(IllegalStateException.class, () -> iterable.iterator());
  }

  public void testFrom_one() {
    Iterable<String> iterable = MinimalIterable.from(singleton("a"));
    Iterator<String> iterator = iterable.iterator();
    assertTrue(iterator.hasNext());
    assertEquals("a", iterator.next());
    assertThrows(UnsupportedOperationException.class, () -> iterator.remove());
    assertFalse(iterator.hasNext());
    assertThrows(NoSuchElementException.class, () -> iterator.next());
    assertThrows(IllegalStateException.class, () -> iterable.iterator());
  }
}
