// Source-based slice around line 405
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_constructorPreferredOverConstants()>


  public void testGet_nonPublicConstantNotUsed() {
    assertNull(ArbitraryInstances.get(NonPublicConstantIgnored.class));
  }

  public void testGet_nonStaticFieldNotUsed() {
    assertNull(ArbitraryInstances.get(NonStaticFieldIgnored.class));
  }

  public void testGet_constructorPreferredOverConstants() {
    assertNotNull(ArbitraryInstances.get(WithPublicConstructorAndConstant.class));
    assertTrue(
        ArbitraryInstances.get(WithPublicConstructorAndConstant.class)
            != ArbitraryInstances.get(WithPublicConstructorAndConstant.class));
  }

  public void testGet_nonFinalFieldNotUsed() {
    assertNull(ArbitraryInstances.get(NonFinalFieldIgnored.class));
  }

