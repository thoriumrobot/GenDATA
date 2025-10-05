// Source-based slice around line 108
// Method: <com.google.common.io.MultiInputStream: long skip(long)>

      if (result != -1) {
        return result;
      }
      advance();
    }
    return -1;
  }

  @Override
  public long skip(long n) throws IOException {
    if (in == null || n <= 0) {
      return 0;
    }
    long result = in.skip(n);
    if (result != 0) {
      return result;
    }
    if (read() == -1) {
      return 0;
    }
