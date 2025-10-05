// Source-based slice around line 93
// Method: com.google.common.hash.Hashing.GOOD_FAST_HASH_SEED

    }
    return new ConcatenatedHashFunction(hashFunctions);
  }

  /**
   * Used to randomize {@link #goodFastHash} instances, so that programs which persist anything
   * dependent on the hash codes they produce will fail sooner.
   */
  @SuppressWarnings("GoodTime") // reading system time without TimeSource
  static final int GOOD_FAST_HASH_SEED = (int) System.currentTimeMillis();

  /**
   * Returns a hash function implementing the <a
   * href="https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp">32-bit murmur3
   * algorithm, x86 variant</a> (little-endian variant), using the given seed value, <b>with a known
   * bug</b> as described in the deprecation text.
   *
   * <p>The C++ equivalent is the MurmurHash3_x86_32 function (Murmur3A), which however does not
   * have the bug.
   *
