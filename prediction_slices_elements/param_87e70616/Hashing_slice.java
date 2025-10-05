// Source-based slice around line 694
// Method: <com.google.common.hash.Hashing: int checkPositiveAndMakeMultipleOf32(int)>

          nextBytes.length == resultBytes.length, "All hashcodes must have the same bit length.");
      for (int i = 0; i < nextBytes.length; i++) {
        resultBytes[i] += nextBytes[i];
      }
    }
    return HashCode.fromBytesNoCopy(resultBytes);
  }

  /** Checks that the passed argument is positive, and ceils it to a multiple of 32. */
  static int checkPositiveAndMakeMultipleOf32(int bits) {
    checkArgument(bits > 0, "Number of bits must be positive");
    return (bits + 31) & ~31;
  }

  /**
   * Returns a hash function which computes its hash code by concatenating the hash codes of the
   * underlying hash functions together. This can be useful if you need to generate hash codes of a
   * specific length.
   *
   * <p>For example, if you need 1024-bit hash codes, you could join two {@link Hashing#sha512} hash
