// Source-based slice around line 40
// Method: <com.google.common.hash.AbstractNonStreamingHashFunction: Hasher newHasher(int)>

 */
@Immutable
abstract class AbstractNonStreamingHashFunction extends AbstractHashFunction {
  @Override
  public Hasher newHasher() {
    return newHasher(32);
  }

  @Override
  public Hasher newHasher(int expectedInputSize) {
    Preconditions.checkArgument(expectedInputSize >= 0);
    return new BufferingHasher(expectedInputSize);
  }

  @Override
  public HashCode hashInt(int input) {
    return hashBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(input).array());
  }

  @Override
