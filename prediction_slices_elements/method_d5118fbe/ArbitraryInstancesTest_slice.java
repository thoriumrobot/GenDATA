// Source-based slice around line 308
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_class()>


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
        ArbitraryInstances.get(WithExceptionalConstructor.class));
    assertNull(ArbitraryInstances.get(NonPublicClass.class));
  }

