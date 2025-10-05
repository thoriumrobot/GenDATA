// Source-based slice around line 59
// Method: <com.google.common.io.LittleEndianDataInputStream: String readLine()>

   */
  public LittleEndianDataInputStream(InputStream in) {
    super(Preconditions.checkNotNull(in));
  }

  /** This method will throw an {@link UnsupportedOperationException}. */
  @CanIgnoreReturnValue // to skip a line
  @Override
  @DoNotCall("Always throws UnsupportedOperationException")
  public String readLine() {
    throw new UnsupportedOperationException("readLine is not supported");
  }

  @Override
  public void readFully(byte[] b) throws IOException {
    ByteStreams.readFully(this, b);
  }

  @Override
  public void readFully(byte[] b, int off, int len) throws IOException {
