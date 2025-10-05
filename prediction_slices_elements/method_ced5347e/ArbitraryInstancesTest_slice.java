// Source-based slice around line 381
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_nullConstantIgnored()>


  public void testGet_usePublicConstant() {
    assertSame(WithPublicConstant.INSTANCE, ArbitraryInstances.get(WithPublicConstant.class));
  }

  public void testGet_useFirstPublicConstant() {
    assertSame(WithPublicConstants.FIRST, ArbitraryInstances.get(WithPublicConstants.class));
  }

  public void testGet_nullConstantIgnored() {
    assertSame(FirstConstantIsNull.SECOND, ArbitraryInstances.get(FirstConstantIsNull.class));
  }

  public void testGet_constantWithGenericsNotUsed() {
    assertNull(ArbitraryInstances.get(WithGenericConstant.class));
  }

  public void testGet_nullConstant() {
    assertNull(ArbitraryInstances.get(WithNullConstant.class));
  }
