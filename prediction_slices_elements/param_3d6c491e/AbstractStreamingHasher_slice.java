// Source-based slice around line 148
// Method: <com.google.common.hash.AbstractStreamingHasher: Hasher putByte(byte)>

   * While intuitively, using CharsetEncoder to encode the CharSequence directly to the buffer (or
   * even to an intermediate buffer) should be considerably more efficient than potentially
   * copying the CharSequence to a String and then calling getBytes(Charset) on that String, in
   * reality there are optimizations that make the getBytes(Charset) approach considerably faster,
   * at least for commonly used charsets like UTF-8.
   */

  @Override
  @CanIgnoreReturnValue
  public final Hasher putByte(byte b) {
    buffer.put(b);
    munchIfFull();
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public final Hasher putShort(short s) {
    buffer.putShort(s);
    munchIfFull();
