// Source-based slice around line 295
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_enum()>

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

  public void testGet_interface() {
    assertNull(ArbitraryInstances.get(SomeInterface.class));
  }

  public void testGet_runnable() {
    ArbitraryInstances.get(Runnable.class).run();
