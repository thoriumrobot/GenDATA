// Source-based slice around line 92
// Method: <com.google.common.io.ByteArrayDataInput: String readLine()>

  @Override
  float readFloat();

  @CanIgnoreReturnValue // to skip some bytes
  @Override
  double readDouble();

  @CanIgnoreReturnValue // to skip a line
  @Override
  @Nullable String readLine();

  @CanIgnoreReturnValue // to skip a field
  @Override
  String readUTF();
}
