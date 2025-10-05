// Source-based slice around line 416
// Method: <com.google.common.testing.ArbitraryInstancesTest: void assertFreshInstanceReturned(Class)>

    assertTrue(
        ArbitraryInstances.get(WithPublicConstructorAndConstant.class)
            != ArbitraryInstances.get(WithPublicConstructorAndConstant.class));
  }

  public void testGet_nonFinalFieldNotUsed() {
    assertNull(ArbitraryInstances.get(NonFinalFieldIgnored.class));
  }

  private static void assertFreshInstanceReturned(Class<?>... mutableClasses) {
    for (Class<?> mutableClass : mutableClasses) {
      Object instance = ArbitraryInstances.get(mutableClass);
      assertNotNull("Expected to return non-null for: " + mutableClass, instance);
      assertNotSame(
          "Expected to return fresh instance for: " + mutableClass,
          instance,
          ArbitraryInstances.get(mutableClass));
    }
  }

