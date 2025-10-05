// Source-based slice around line 368
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_regex()>

    assertNotNull(ArbitraryInstances.get(CharSink.class));
  }

  public void testGet_reflect() {
    assertNotNull(ArbitraryInstances.get(Type.class));
    assertNotNull(ArbitraryInstances.get(AnnotatedElement.class));
    assertNotNull(ArbitraryInstances.get(GenericDeclaration.class));
  }

  public void testGet_regex() {
    assertEquals(Pattern.compile("").pattern(), ArbitraryInstances.get(Pattern.class).pattern());
    assertEquals(0, ArbitraryInstances.get(MatchResult.class).groupCount());
  }

  public void testGet_usePublicConstant() {
    assertSame(WithPublicConstant.INSTANCE, ArbitraryInstances.get(WithPublicConstant.class));
  }

  public void testGet_useFirstPublicConstant() {
    assertSame(WithPublicConstants.FIRST, ArbitraryInstances.get(WithPublicConstants.class));
