// Source-based slice around line 281
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_comparable()>

  @SuppressWarnings("unchecked") // functor classes have no type parameters
  public void testGet_functors() {
    assertEquals(0, ArbitraryInstances.get(Comparator.class).compare("abc", 123));
    assertTrue(ArbitraryInstances.get(Predicate.class).apply("abc"));
    assertTrue(ArbitraryInstances.get(Equivalence.class).equivalent(1, 1));
    assertFalse(ArbitraryInstances.get(Equivalence.class).equivalent(1, 2));
  }

  @SuppressWarnings("SelfComparison")
  public void testGet_comparable() {
    @SuppressWarnings("unchecked") // The null value can compare with any Object
    Comparable<Object> comparable = ArbitraryInstances.get(Comparable.class);
    assertEquals(0, comparable.compareTo(comparable));
    assertTrue(comparable.compareTo("") > 0);
    assertThrows(NullPointerException.class, () -> comparable.compareTo(null));
  }

  public void testGet_array() {
    assertThat(ArbitraryInstances.get(int[].class)).isEmpty();
    assertThat(ArbitraryInstances.get(Object[].class)).isEmpty();
