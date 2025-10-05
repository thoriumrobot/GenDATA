// Source-based slice around line 79
// Method: <com.google.common.cache.ReferenceEntry: ReferenceEntry getPreviousInAccessQueue()>

  void setAccessTime(long time);

  /** Returns the next entry in the access queue. */
  ReferenceEntry<K, V> getNextInAccessQueue();

  /** Sets the next entry in the access queue. */
  void setNextInAccessQueue(ReferenceEntry<K, V> next);

  /** Returns the previous entry in the access queue. */
  ReferenceEntry<K, V> getPreviousInAccessQueue();

  /** Sets the previous entry in the access queue. */
  void setPreviousInAccessQueue(ReferenceEntry<K, V> previous);

  /*
   * Implemented by entries that use write order. Write entries are maintained in a doubly-linked
   * list. New entries are added at the tail of the list at write time and stale entries are
   * expired from the head of the list.
   */

