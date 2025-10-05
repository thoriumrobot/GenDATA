// Source-based slice around line 393
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_constantTypeDoesNotMatch()>


  public void testGet_constantWithGenericsNotUsed() {
    assertNull(ArbitraryInstances.get(WithGenericConstant.class));
  }

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
