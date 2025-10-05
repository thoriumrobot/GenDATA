// Source-based slice around line 118
// Method: <com.google.common.io.ReaderInputStream: void close()>

    encoder.reset();

    charBuffer = CharBuffer.allocate(bufferSize);
    Java8Compatibility.flip(charBuffer);

    byteBuffer = ByteBuffer.allocate(bufferSize);
  }

  @Override
  public void close() throws IOException {
    reader.close();
  }

  @Override
  public int read() throws IOException {
    return (read(singleByte) == 1) ? toUnsignedInt(singleByte[0]) : -1;
  }

  // TODO(chrisn): Consider trying to encode/flush directly to the argument byte
  // buffer when possible.
