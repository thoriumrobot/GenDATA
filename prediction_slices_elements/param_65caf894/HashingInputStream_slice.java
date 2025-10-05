// Source-based slice around line 66
// Method: <com.google.common.hash.HashingInputStream: int read(byte[],int,int)>

    return b;
  }

  /**
   * Reads the specified bytes of data from the underlying input stream and updates the hasher with
   * the bytes read.
   */
  @Override
  @CanIgnoreReturnValue
  public int read(byte[] bytes, int off, int len) throws IOException {
    int numOfBytesRead = in.read(bytes, off, len);
    if (numOfBytesRead != -1) {
      hasher.putBytes(bytes, off, numOfBytesRead);
    }
    return numOfBytesRead;
  }

  /**
   * mark() is not supported for HashingInputStream
   *
