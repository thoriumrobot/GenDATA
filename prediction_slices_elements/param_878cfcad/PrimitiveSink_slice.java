// Source-based slice around line 37
// Method: <com.google.common.hash.PrimitiveSink: PrimitiveSink putByte(byte)>

@Beta
public interface PrimitiveSink {
  /**
   * Puts a byte into this sink.
   *
   * @param b a byte
   * @return this instance
   */
  @CanIgnoreReturnValue
  PrimitiveSink putByte(byte b);

  /**
   * Puts an array of bytes into this sink.
   *
   * @param bytes a byte array
   * @return this instance
   */
  @CanIgnoreReturnValue
  PrimitiveSink putBytes(byte[] bytes);

