// Source-based slice around line 35
// Method: <com.google.common.cache.Weigher: int weigh(K,V)>

@FunctionalInterface
public interface Weigher<K, V> {

  /**
   * Returns the weight of a cache entry. There is no unit for entry weights; rather they are simply
   * relative to each other.
   *
   * @return the weight of the entry; must be non-negative
   */
  int weigh(K key, V value);
}
