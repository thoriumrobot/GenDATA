// Source-based slice around line 92
// Method: <com.google.common.cache.ReferenceEntry: long getWriteTime()>


  /*
   * Implemented by entries that use write order. Write entries are maintained in a doubly-linked
   * list. New entries are added at the tail of the list at write time and stale entries are
   * expired from the head of the list.
   */

  /** Returns the time that this entry was last written, in ns. */
  @SuppressWarnings("GoodTime")
  long getWriteTime();

  /** Sets the entry write time in ns. */
  @SuppressWarnings("GoodTime") // b/122668874
  void setWriteTime(long time);

  /** Returns the next entry in the write queue. */
  ReferenceEntry<K, V> getNextInWriteQueue();

  /** Sets the next entry in the write queue. */
  void setNextInWriteQueue(ReferenceEntry<K, V> next);
