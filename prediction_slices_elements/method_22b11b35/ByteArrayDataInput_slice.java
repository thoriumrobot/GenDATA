// Source-based slice around line 80
// Method: <com.google.common.io.ByteArrayDataInput: long readLong()>

  @Override
  char readChar();

  @CanIgnoreReturnValue // to skip some bytes
  @Override
  int readInt();

  @CanIgnoreReturnValue // to skip some bytes
  @Override
  long readLong();

  @CanIgnoreReturnValue // to skip some bytes
  @Override
  float readFloat();

  @CanIgnoreReturnValue // to skip some bytes
  @Override
  double readDouble();

  @CanIgnoreReturnValue // to skip a line
