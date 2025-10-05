// Source-based slice around line 63
// Method: <com.google.common.io.ByteArrayDataOutput: void writeDouble(double)>

  void writeInt(int v);

  @Override
  void writeLong(long v);

  @Override
  void writeFloat(float v);

  @Override
  void writeDouble(double v);

  @Override
  void writeChars(String s);

  @Override
  void writeUTF(String s);

  /**
   * @deprecated This method is dangerous as it discards the high byte of every character. For
   *     UTF-8, use {@code write(s.getBytes(StandardCharsets.UTF_8))}.
