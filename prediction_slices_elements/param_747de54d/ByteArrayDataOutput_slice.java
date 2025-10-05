// Source-based slice around line 69
// Method: <com.google.common.io.ByteArrayDataOutput: void writeUTF(String)>

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
   */
  @Deprecated
  @Override
  void writeBytes(String s);

  /** Returns the contents that have been written to this instance, as a byte array. */
