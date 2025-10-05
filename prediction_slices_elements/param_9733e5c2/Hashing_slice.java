// Source-based slice around line 710
// Method: <com.google.common.hash.Hashing: HashFunction concatenating(HashFunction,HashFunction,HashFunction)>

   * underlying hash functions together. This can be useful if you need to generate hash codes of a
   * specific length.
   *
   * <p>For example, if you need 1024-bit hash codes, you could join two {@link Hashing#sha512} hash
   * functions together: {@code Hashing.concatenating(Hashing.sha512(), Hashing.sha512())}.
   *
   * @since 19.0
   */
  public static HashFunction concatenating(
      HashFunction first, HashFunction second, HashFunction... rest) {
    // We can't use Lists.asList() here because there's no hash->collect dependency
    List<HashFunction> list = new ArrayList<>();
    list.add(first);
    list.add(second);
    Collections.addAll(list, rest);
    return new ConcatenatedHashFunction(list.toArray(new HashFunction[0]));
  }

  /**
   * Returns a hash function which computes its hash code by concatenating the hash codes of the
