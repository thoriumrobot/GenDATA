// Source-based slice around line 102
// Method: <com.google.common.cache.ReferenceEntry: void setNextInWriteQueue(ReferenceEntry)>


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
