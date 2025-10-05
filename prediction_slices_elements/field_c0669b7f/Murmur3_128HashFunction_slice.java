// Source-based slice around line 215
// Method: com.google.common.hash.Murmur3_128HashFunction.serialVersionUID


    private static long mixK2(long k2) {
      k2 *= C2;
      k2 = Long.rotateLeft(k2, 33);
      k2 *= C1;
      return k2;
    }
  }

  private static final long serialVersionUID = 0L;
}
