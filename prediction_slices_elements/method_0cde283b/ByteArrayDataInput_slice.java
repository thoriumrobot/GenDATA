// Source-based slice around line 48
// Method: <com.google.common.io.ByteArrayDataInput: int skipBytes(int)>

  @Override
  void readFully(byte[] b);

  @Override
  void readFully(byte[] b, int off, int len);

  // not guaranteed to skip n bytes so result should NOT be ignored
  // use ByteStreams.skipFully or one of the read methods instead
  @Override
  int skipBytes(int n);

  @CanIgnoreReturnValue // to skip a byte
  @Override
  boolean readBoolean();

  @CanIgnoreReturnValue // to skip a byte
  @Override
  byte readByte();

  @CanIgnoreReturnValue // to skip a byte
