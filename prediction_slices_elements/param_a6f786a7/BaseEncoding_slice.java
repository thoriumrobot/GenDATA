// Source-based slice around line 266
// Method: <com.google.common.io.BaseEncoding: void encodeTo(Appendable,byte[],int,int)>

        return decodingStream(encodedSource.openStream());
      }
    };
  }

  // Implementations for encoding/decoding

  abstract int maxEncodedSize(int bytes);

  abstract void encodeTo(Appendable target, byte[] bytes, int off, int len) throws IOException;

  abstract int maxDecodedSize(int chars);

  abstract int decodeTo(byte[] target, CharSequence chars) throws DecodingException;

  CharSequence trimTrailingPadding(CharSequence chars) {
    return checkNotNull(chars);
  }

  // Modified encoding generators
