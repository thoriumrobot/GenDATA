// Source-based slice around line 50
// Method: <com.google.common.cache.ReferenceEntry: ReferenceEntry getNext()>

@GwtIncompatible
interface ReferenceEntry<K, V> {
  /** Returns the value reference from this entry. */
  @Nullable ValueReference<K, V> getValueReference();

  /** Sets the value reference for this entry. */
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
