// Source-based slice around line 36
// Method: <com.google.common.io.ByteArrayDataOutput: void write(byte[])>

 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
public interface ByteArrayDataOutput extends DataOutput {
  @Override
  void write(int b);

  @Override
  void write(byte[] b);

  @Override
  void write(byte[] b, int off, int len);

  @Override
  void writeBoolean(boolean v);

  @Override
  void writeByte(int v);

