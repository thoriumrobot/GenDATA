// Source-based slice around line 191
// Method: <com.google.common.io.BaseEncoding: byte[] extract(byte[],int)>

      @Override
      public OutputStream openStream() throws IOException {
        return encodingStream(encodedSink.openStream());
      }
    };
  }

  // TODO(lowasser): document the extent of leniency, probably after adding ignore(CharMatcher)

  private static byte[] extract(byte[] result, int length) {
    if (length == result.length) {
      return result;
    }
    byte[] trunc = new byte[length];
    System.arraycopy(result, 0, trunc, 0, length);
    return trunc;
  }

  /**
   * Determines whether the specified character sequence is a valid encoded string according to this
