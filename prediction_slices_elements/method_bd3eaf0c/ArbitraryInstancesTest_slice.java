// Source-based slice around line 377
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_useFirstPublicConstant()>

  public void testGet_regex() {
    assertEquals(Pattern.compile("").pattern(), ArbitraryInstances.get(Pattern.class).pattern());
    assertEquals(0, ArbitraryInstances.get(MatchResult.class).groupCount());
  }

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
