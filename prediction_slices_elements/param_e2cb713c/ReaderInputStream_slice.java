// Source-based slice around line 255
// Method: <com.google.common.io.ReaderInputStream: int drain(byte[],int,int)>

    } else {
      draining = true;
    }
  }

  /**
   * Copy as much of the byte buffer into the output array as possible, returning the (positive)
   * number of characters copied.
   */
  private int drain(byte[] b, int off, int len) {
    int remaining = min(len, byteBuffer.remaining());
    byteBuffer.get(b, off, remaining);
    return remaining;
  }
}
