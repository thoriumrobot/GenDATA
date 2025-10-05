// Source-based slice around line 71
// Method: <com.google.common.hash.AbstractNonStreamingHashFunction: HashCode hashBytes(byte[],int,int)>

    return hashBytes(buffer.array());
  }

  @Override
  public HashCode hashString(CharSequence input, Charset charset) {
    return hashBytes(input.toString().getBytes(charset));
  }

  @Override
  public abstract HashCode hashBytes(byte[] input, int off, int len);

  @Override
  public HashCode hashBytes(ByteBuffer input) {
    return newHasher(input.remaining()).putBytes(input).hash();
  }

  /** In-memory stream-based implementation of Hasher. */
  private final class BufferingHasher extends AbstractHasher {
    final ExposedByteArrayOutputStream stream;

