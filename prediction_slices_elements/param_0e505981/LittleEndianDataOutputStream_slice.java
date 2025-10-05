// Source-based slice around line 51
// Method: <com.google.common.io.LittleEndianDataOutputStream: void write(byte[],int,int)>

   * Creates a {@code LittleEndianDataOutputStream} that wraps the given stream.
   *
   * @param out the stream to delegate to
   */
  public LittleEndianDataOutputStream(OutputStream out) {
    super(new DataOutputStream(Preconditions.checkNotNull(out)));
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
