// Source-based slice around line 176
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_collections()>

    assertEquals("", ArbitraryInstances.get(String.class));
    assertEquals("", ArbitraryInstances.get(CharSequence.class));
    assertEquals(SECONDS, ArbitraryInstances.get(TimeUnit.class));
    assertNotNull(ArbitraryInstances.get(Object.class));
    assertEquals(0, ArbitraryInstances.get(Number.class));
    assertEquals(UTF_8, ArbitraryInstances.get(Charset.class));
    assertNotNull(ArbitraryInstances.get(UUID.class));
  }

  public void testGet_collections() {
    assertEquals(ImmutableSet.of().iterator(), ArbitraryInstances.get(Iterator.class));
    assertFalse(ArbitraryInstances.get(PeekingIterator.class).hasNext());
    assertFalse(ArbitraryInstances.get(ListIterator.class).hasNext());
    assertEquals(ImmutableSet.of(), ArbitraryInstances.get(Iterable.class));
    assertEquals(ImmutableSet.of(), ArbitraryInstances.get(Set.class));
    assertEquals(ImmutableSet.of(), ArbitraryInstances.get(ImmutableSet.class));
    assertEquals(ImmutableSortedSet.of(), ArbitraryInstances.get(SortedSet.class));
    assertEquals(ImmutableSortedSet.of(), ArbitraryInstances.get(ImmutableSortedSet.class));
    assertEquals(ImmutableList.of(), ArbitraryInstances.get(Collection.class));
    assertEquals(ImmutableList.of(), ArbitraryInstances.get(ImmutableCollection.class));
