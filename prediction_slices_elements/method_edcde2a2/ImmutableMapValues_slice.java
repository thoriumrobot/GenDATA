// Source-based slice around line 45
// Method: <com.google.common.collect.ImmutableMapValues: int size()>

@GwtCompatible
final class ImmutableMapValues<K, V> extends ImmutableCollection<V> {
  private final ImmutableMap<K, V> map;

  ImmutableMapValues(ImmutableMap<K, V> map) {
    this.map = map;
  }

  @Override
  public int size() {
    return map.size();
  }

  @Override
  public UnmodifiableIterator<V> iterator() {
    return new UnmodifiableIterator<V>() {
      final UnmodifiableIterator<Entry<K, V>> entryItr = map.entrySet().iterator();

      @Override
      public boolean hasNext() {
