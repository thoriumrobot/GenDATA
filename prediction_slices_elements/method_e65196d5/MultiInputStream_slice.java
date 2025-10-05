// Source-based slice around line 78
// Method: <com.google.common.io.MultiInputStream: boolean markSupported()>

  @Override
  public int available() throws IOException {
    if (in == null) {
      return 0;
    }
    return in.available();
  }

  @Override
  public boolean markSupported() {
    return false;
  }

  @Override
  public int read() throws IOException {
    while (in != null) {
      int result = in.read();
      if (result != -1) {
        return result;
      }
