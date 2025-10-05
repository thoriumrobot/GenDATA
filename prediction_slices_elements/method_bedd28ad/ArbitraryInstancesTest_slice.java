// Source-based slice around line 304
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_runnable()>

  public void testGet_enum() {
    assertNull(ArbitraryInstances.get(EmptyEnum.class));
    assertEquals(Direction.UP, ArbitraryInstances.get(Direction.class));
  }

  public void testGet_interface() {
    assertNull(ArbitraryInstances.get(SomeInterface.class));
  }

  public void testGet_runnable() {
    ArbitraryInstances.get(Runnable.class).run();
  }

  public void testGet_class() {
    assertSame(SomeAbstractClass.INSTANCE, ArbitraryInstances.get(SomeAbstractClass.class));
    assertSame(
        WithPrivateConstructor.INSTANCE, ArbitraryInstances.get(WithPrivateConstructor.class));
    assertNull(ArbitraryInstances.get(NoDefaultConstructor.class));
    assertSame(
        WithExceptionalConstructor.INSTANCE,
