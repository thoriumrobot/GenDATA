// Source-based slice around line 729
// Method: <com.google.common.hash.Hashing: HashFunction concatenating(Iterable)>

   * Returns a hash function which computes its hash code by concatenating the hash codes of the
   * underlying hash functions together. This can be useful if you need to generate hash codes of a
   * specific length.
   *
   * <p>For example, if you need 1024-bit hash codes, you could join two {@link Hashing#sha512} hash
   * functions together: {@code Hashing.concatenating(Hashing.sha512(), Hashing.sha512())}.
   *
   * @since 19.0
   */
  public static HashFunction concatenating(Iterable<HashFunction> hashFunctions) {
    checkNotNull(hashFunctions);
    // We can't use Iterables.toArray() here because there's no hash->collect dependency
    List<HashFunction> list = new ArrayList<>();
    for (HashFunction hashFunction : hashFunctions) {
      list.add(hashFunction);
    }
    checkArgument(!list.isEmpty(), "number of hash functions (%s) must be > 0", list.size());
    return new ConcatenatedHashFunction(list.toArray(new HashFunction[0]));
  }

