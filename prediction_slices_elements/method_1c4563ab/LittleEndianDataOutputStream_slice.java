// Source-based slice around line 62
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeByte(int)>

    out.write(b, off, len);
  }

  @Override
  public void writeBoolean(boolean v) throws IOException {
    ((DataOutputStream) out).writeBoolean(v);
  }

  @Override
  public void writeByte(int v) throws IOException {
    ((DataOutputStream) out).writeByte(v);
  }

  /**
   * @deprecated The semantics of {@code writeBytes(String s)} are considered dangerous. Please use
   *     {@link #writeUTF(String s)}, {@link #writeChars(String s)} or another write method instead.
   */
  @Deprecated
  @Override
  public void writeBytes(String s) throws IOException {
