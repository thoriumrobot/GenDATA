// Source-based slice around line 80
// Method: <com.google.common.io.LittleEndianDataInputStream: int readUnsignedByte()>

  }

  @Override
  public int skipBytes(int n) throws IOException {
    return (int) in.skip(n);
  }

  @CanIgnoreReturnValue // to skip a byte
  @Override
  public int readUnsignedByte() throws IOException {
    int b1 = in.read();
    if (b1 < 0) {
      throw new EOFException();
    }

    return b1;
  }

  /**
   * Reads an unsigned {@code short} as specified by {@link DataInputStream#readUnsignedShort()},
