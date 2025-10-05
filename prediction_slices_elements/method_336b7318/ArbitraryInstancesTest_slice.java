// Source-based slice around line 289
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_array()>

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
    assertThat(ArbitraryInstances.get(String[].class)).isEmpty();
  }

  public void testGet_enum() {
    assertNull(ArbitraryInstances.get(EmptyEnum.class));
    assertEquals(Direction.UP, ArbitraryInstances.get(Direction.class));
  }

