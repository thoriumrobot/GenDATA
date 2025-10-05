// Source-based slice around line 1086
// Method: <com.google.common.collect.MapMakerInternalMap: int rehash(int)>


  /**
   * Applies a supplemental hash function to a given hash code, which defends against poor quality
   * hash functions. This is critical when the concurrent hash map uses power-of-two length hash
   * tables, that otherwise encounter collisions for hash codes that do not differ in lower or upper
   * bits.
   *
   * @param h hash code
   */
  static int rehash(int h) {
    // Spread bits to regularize both segment and index locations,
    // using variant of single-word Wang/Jenkins hash.
    // TODO(kevinb): use Hashing/move this to Hashing?
    h += (h << 15) ^ 0xffffcd7d;
    h ^= h >>> 10;
    h += h << 3;
    h ^= h >>> 6;
    h += (h << 2) + (h << 14);
    return h ^ (h >>> 16);
  }
