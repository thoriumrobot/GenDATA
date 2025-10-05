// Source-based slice around line 84
// Method: <com.google.common.io.ByteArrayDataInput: float readFloat()>

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
  @Override
  @Nullable String readLine();

  @CanIgnoreReturnValue // to skip a field
