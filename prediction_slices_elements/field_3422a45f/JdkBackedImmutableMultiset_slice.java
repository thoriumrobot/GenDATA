// Source-based slice around line 39
// Method: com.google.common.collect.JdkBackedImmutableMultiset.size

 * An implementation of ImmutableMultiset backed by a JDK Map and a list of entries. Used to protect
 * against hash flooding attacks.
 *
 * @author Louis Wasserman
 */
@GwtCompatible
final class JdkBackedImmutableMultiset<E> extends ImmutableMultiset<E> {
  private final Map<E, Integer> delegateMap;
  private final ImmutableList<Entry<E>> entries;
  private final long size;

  static <E> ImmutableMultiset<E> create(Collection<? extends Entry<? extends E>> entries) {
    @SuppressWarnings("unchecked")
    Entry<E>[] entriesArray = entries.toArray((Entry<E>[]) new Entry<?>[0]);
    Map<E, Integer> delegateMap = Maps.newHashMapWithExpectedSize(entriesArray.length);
    long size = 0;
    for (int i = 0; i < entriesArray.length; i++) {
      Entry<E> entry = entriesArray[i];
      int count = entry.getCount();
      size += count;
