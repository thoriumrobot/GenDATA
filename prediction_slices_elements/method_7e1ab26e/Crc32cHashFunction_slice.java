// Source-based slice around line 41
// Method: <com.google.common.hash.Crc32cHashFunction: String toString()>

    return 32;
  }

  @Override
  public Hasher newHasher() {
    return new Crc32cHasher();
  }

  @Override
  public String toString() {
    return "Hashing.crc32c()";
  }

  static final class Crc32cHasher extends AbstractStreamingHasher {

    /*
     * The striding algorithm works roughly as follows: it is universally the case that
     * CRC(x ^ y) == CRC(x) ^ CRC(y).  The approach we take is to break the message as follows,
     * with each letter representing a 4-byte word: ABCDABCDABCDABCD... and to calculate
     * CRC(A000A000A000...), CRC(0B000B000B...), CRC(00C000C000C...), CRC(000D000D000D...)
