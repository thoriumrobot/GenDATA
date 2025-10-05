// Source-based slice around line 108
// Method: <com.google.common.cache.ReferenceEntry: void setPreviousInWriteQueue(ReferenceEntry)>

  ReferenceEntry<K, V> getNextInWriteQueue();

  /** Sets the next entry in the write queue. */
  void setNextInWriteQueue(ReferenceEntry<K, V> next);

  /** Returns the previous entry in the write queue. */
  ReferenceEntry<K, V> getPreviousInWriteQueue();

  /** Sets the previous entry in the write queue. */
  void setPreviousInWriteQueue(ReferenceEntry<K, V> previous);
}
