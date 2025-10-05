// Source-based slice around line 401
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_nonStaticFieldNotUsed()>


  public void testGet_constantTypeDoesNotMatch() {
    assertNull(ArbitraryInstances.get(ParentClassHasConstant.class));
  }

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

