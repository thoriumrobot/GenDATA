// Source-based slice around line 44
// Method: <com.google.common.cache.ReferenceEntry: ValueReference getValueReference()>

 * <ul>
 *   <li>Expired: time expired (key/value may still be set)
 *   <li>Collected: key/value was partially collected, but not yet cleaned up
 *   <li>Unset: marked as unset, awaiting cleanup or reuse
 * </ul>
 */
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

