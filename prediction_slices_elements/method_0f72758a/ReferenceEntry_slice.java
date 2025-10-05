// Source-based slice around line 56
// Method: <com.google.common.cache.ReferenceEntry: K getKey()>

  void setValueReference(ValueReference<K, V> valueReference);

  /** Returns the next entry in the chain. */
  @Nullable ReferenceEntry<K, V> getNext();

  /** Returns the entry's hash. */
  int getHash();

  /** Returns the key for this entry. */
  @Nullable K getKey();

  /*
   * Used by entries that use access order. Access entries are maintained in a doubly-linked list.
   * New entries are added at the tail of the list at write time; stale entries are expired from
   * the head of the list.
   */

  /** Returns the time that this entry was last accessed, in ns. */
  @SuppressWarnings("GoodTime")
  long getAccessTime();
