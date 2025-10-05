// Source-based slice around line 83
// Method: <com.google.common.io.MultiInputStream: int read()>

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
      advance();
    }
    return -1;
  }

