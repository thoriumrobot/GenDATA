// Source-based slice around line 67
// Method: <com.google.common.io.MultiReader: long skip(long)>

    int result = current.read(cbuf, off, len);
    if (result == -1) {
      advance();
      return read(cbuf, off, len);
    }
    return result;
  }

  @Override
  public long skip(long n) throws IOException {
    Preconditions.checkArgument(n >= 0, "n is negative");
    if (n > 0) {
      while (current != null) {
        long result = current.skip(n);
        if (result > 0) {
          return result;
        }
        advance();
      }
    }
