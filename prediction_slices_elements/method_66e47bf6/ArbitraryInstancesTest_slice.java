// Source-based slice around line 397
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_nonPublicConstantNotUsed()>


  public void testGet_nullConstant() {
    assertNull(ArbitraryInstances.get(WithNullConstant.class));
  }

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
