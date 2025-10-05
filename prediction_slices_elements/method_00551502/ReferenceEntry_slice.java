// Source-based slice around line 99
// Method: <com.google.common.cache.ReferenceEntry: ReferenceEntry getNextInWriteQueue()>

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

  /** Returns the previous entry in the write queue. */
  ReferenceEntry<K, V> getPreviousInWriteQueue();

  /** Sets the previous entry in the write queue. */
  void setPreviousInWriteQueue(ReferenceEntry<K, V> previous);
}
