// Source-based slice around line 319
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_mutable()>

    assertSame(
        WithPrivateConstructor.INSTANCE, ArbitraryInstances.get(WithPrivateConstructor.class));
    assertNull(ArbitraryInstances.get(NoDefaultConstructor.class));
    assertSame(
        WithExceptionalConstructor.INSTANCE,
        ArbitraryInstances.get(WithExceptionalConstructor.class));
    assertNull(ArbitraryInstances.get(NonPublicClass.class));
  }

  public void testGet_mutable() {
    assertEquals(0, ArbitraryInstances.get(ArrayList.class).size());
    assertEquals(0, ArbitraryInstances.get(HashMap.class).size());
    assertThat(ArbitraryInstances.get(Appendable.class).toString()).isEmpty();
    assertThat(ArbitraryInstances.get(StringBuilder.class).toString()).isEmpty();
    assertThat(ArbitraryInstances.get(StringBuffer.class).toString()).isEmpty();
    assertFreshInstanceReturned(
        ArrayList.class,
        HashMap.class,
        Appendable.class,
        StringBuilder.class,
