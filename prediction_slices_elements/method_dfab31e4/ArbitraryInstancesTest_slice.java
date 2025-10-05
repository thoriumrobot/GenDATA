// Source-based slice around line 362
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_reflect()>

        ByteArrayOutputStream.class, OutputStream.class,
        Writer.class, StringWriter.class,
        PrintStream.class, PrintWriter.class);
    assertEquals(ByteSource.empty(), ArbitraryInstances.get(ByteSource.class));
    assertEquals(CharSource.empty(), ArbitraryInstances.get(CharSource.class));
    assertNotNull(ArbitraryInstances.get(ByteSink.class));
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

