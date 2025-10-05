// Source-based slice around line 389
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_nullConstant()>


  public void testGet_nullConstantIgnored() {
    assertSame(FirstConstantIsNull.SECOND, ArbitraryInstances.get(FirstConstantIsNull.class));
  }

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
