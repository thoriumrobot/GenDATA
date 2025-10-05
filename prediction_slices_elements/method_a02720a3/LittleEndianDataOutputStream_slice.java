// Source-based slice around line 57
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeBoolean(boolean)>

  }

  @Override
  public void write(byte[] b, int off, int len) throws IOException {
    // Override slow FilterOutputStream impl
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
