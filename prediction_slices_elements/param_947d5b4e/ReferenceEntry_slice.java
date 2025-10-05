// Source-based slice around line 70
// Method: <com.google.common.cache.ReferenceEntry: void setAccessTime(long)>

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

  /** Returns the previous entry in the access queue. */
  ReferenceEntry<K, V> getPreviousInAccessQueue();

