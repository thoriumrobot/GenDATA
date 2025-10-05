// Source-based slice around line 45
// Method: com.google.common.graph.MapIteratorCache.backingMap

 * correctness in the face of external mutations to the backing map. As such, it is <b>strongly</b>
 * recommended that the caller does not persist a reference to the backing map (unless the backing
 * map is immutable).
 *
 * <p>This class is tailored toward use cases in common.graph. It is *NOT* a general purpose map.
 *
 * @author James Sexton
 */
class MapIteratorCache<K, V> {
  private final Map<K, V> backingMap;

  /*
   * Per JDK: "the behavior of a map entry is undefined if the backing map has been modified after
   * the entry was returned by the iterator, except through the setValue operation on the map entry"
   * As such, this field must be cleared before every map mutation.
   *
   * Note about volatile: volatile doesn't make it safe to read from a mutable graph in one thread
   * while writing to it in another. All it does is help with _reading_ from multiple threads
   * concurrently. For more information, see AbstractNetworkTest.concurrentIteration.
   */
