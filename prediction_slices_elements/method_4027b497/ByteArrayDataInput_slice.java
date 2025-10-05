// Source-based slice around line 64
// Method: <com.google.common.io.ByteArrayDataInput: short readShort()>

  @Override
  byte readByte();

  @CanIgnoreReturnValue // to skip a byte
  @Override
  int readUnsignedByte();

  @CanIgnoreReturnValue // to skip some bytes
  @Override
  short readShort();

  @CanIgnoreReturnValue // to skip some bytes
  @Override
  int readUnsignedShort();

  @CanIgnoreReturnValue // to skip some bytes
  @Override
  char readChar();

  @CanIgnoreReturnValue // to skip some bytes
