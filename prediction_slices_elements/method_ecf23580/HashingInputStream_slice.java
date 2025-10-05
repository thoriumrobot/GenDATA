// Source-based slice around line 80
// Method: <com.google.common.hash.HashingInputStream: boolean markSupported()>

    return numOfBytesRead;
  }

  /**
   * mark() is not supported for HashingInputStream
   *
   * @return {@code false} always
   */
  @Override
  public boolean markSupported() {
    return false;
  }

  /** mark() is not supported for HashingInputStream */
  @Override
  public void mark(int readlimit) {}

  /**
   * reset() is not supported for HashingInputStream.
   *
