// Source-based slice around line 66
// Method: <com.google.common.cache.ReferenceEntry: long getAccessTime()>


  /*
   * Used by entries that use access order. Access entries are maintained in a doubly-linked list.
   * New entries are added at the tail of the list at write time; stale entries are expired from
   * the head of the list.
   */

  /** Returns the time that this entry was last accessed, in ns. */
  @SuppressWarnings("GoodTime")
  long getAccessTime();

  /** Sets the entry access time in ns. */
  @SuppressWarnings("GoodTime") // b/122668874
  void setAccessTime(long time);

  /** Returns the next entry in the access queue. */
  ReferenceEntry<K, V> getNextInAccessQueue();

  /** Sets the next entry in the access queue. */
  void setNextInAccessQueue(ReferenceEntry<K, V> next);
