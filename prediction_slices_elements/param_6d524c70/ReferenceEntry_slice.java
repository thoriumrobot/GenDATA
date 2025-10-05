// Source-based slice around line 47
// Method: <com.google.common.cache.ReferenceEntry: void setValueReference(ValueReference)>

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

  /** Returns the key for this entry. */
  @Nullable K getKey();

