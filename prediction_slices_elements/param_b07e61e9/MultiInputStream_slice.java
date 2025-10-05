// Source-based slice around line 95
// Method: <com.google.common.io.MultiInputStream: int read(byte[],int,int)>

      if (result != -1) {
        return result;
      }
      advance();
    }
    return -1;
  }

  @Override
  public int read(byte[] b, int off, int len) throws IOException {
    checkNotNull(b);
    while (in != null) {
      int result = in.read(b, off, len);
      if (result != -1) {
        return result;
      }
      advance();
    }
    return -1;
  }
