// Source-based slice around line 412
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder initialCapacity(int)>

   * having a hash table of size eight. Providing a large enough estimate at construction time
   * avoids the need for expensive resizing operations later, but setting this value unnecessarily
   * high wastes memory.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalArgumentException if {@code initialCapacity} is negative
   * @throws IllegalStateException if an initial capacity was already set
   */
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> initialCapacity(int initialCapacity) {
    checkState(
        this.initialCapacity == UNSET_INT,
        "initial capacity was already set to %s",
        this.initialCapacity);
    checkArgument(initialCapacity >= 0);
    this.initialCapacity = initialCapacity;
    return this;
  }

  int getInitialCapacity() {
