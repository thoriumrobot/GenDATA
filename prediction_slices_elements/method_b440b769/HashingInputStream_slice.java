// Source-based slice around line 52
// Method: <com.google.common.hash.HashingInputStream: int read()>

    this.hasher = checkNotNull(hashFunction.newHasher());
  }

  /**
   * Reads the next byte of data from the underlying input stream and updates the hasher with the
   * byte read.
   */
  @Override
  @CanIgnoreReturnValue
  public int read() throws IOException {
    int b = in.read();
    if (b != -1) {
      hasher.putByte((byte) b);
    }
    return b;
  }

  /**
   * Reads the specified bytes of data from the underlying input stream and updates the hasher with
   * the bytes read.
