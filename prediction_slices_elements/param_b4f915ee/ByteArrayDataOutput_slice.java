// Source-based slice around line 39
// Method: <com.google.common.io.ByteArrayDataOutput: void write(byte[],int,int)>

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

  @Override
  void writeShort(int v);

