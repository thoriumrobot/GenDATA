// Source-based slice around line 335
// Method: <com.google.common.testing.ArbitraryInstancesTest: void testGet_io()>

        ArrayList.class,
        HashMap.class,
        Appendable.class,
        StringBuilder.class,
        StringBuffer.class,
        Throwable.class,
        Exception.class);
  }

  public void testGet_io() throws IOException {
    assertEquals(-1, ArbitraryInstances.get(InputStream.class).read());
    assertEquals(-1, ArbitraryInstances.get(ByteArrayInputStream.class).read());
    assertEquals(-1, ArbitraryInstances.get(Readable.class).read(CharBuffer.allocate(1)));
    assertEquals(-1, ArbitraryInstances.get(Reader.class).read());
    assertEquals(-1, ArbitraryInstances.get(StringReader.class).read());
    assertEquals(0, ArbitraryInstances.get(Buffer.class).capacity());
    assertEquals(0, ArbitraryInstances.get(CharBuffer.class).capacity());
    assertEquals(0, ArbitraryInstances.get(ByteBuffer.class).capacity());
    assertEquals(0, ArbitraryInstances.get(ShortBuffer.class).capacity());
    assertEquals(0, ArbitraryInstances.get(IntBuffer.class).capacity());
