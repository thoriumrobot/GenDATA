// Source-based slice around line 42
// Method: com.google.common.collect.SingletonImmutableBiMap.singleValue

 *
 * @author Jesse Wilson
 * @author Kevin Bourrillion
 */
@GwtCompatible
@SuppressWarnings("serial") // uses writeReplace(), not default serialization
final class SingletonImmutableBiMap<K, V> extends ImmutableBiMap<K, V> {

  final transient K singleKey;
  final transient V singleValue;

  SingletonImmutableBiMap(K singleKey, V singleValue) {
    checkEntryNotNull(singleKey, singleValue);
    this.singleKey = singleKey;
    this.singleValue = singleValue;
    this.inverse = null;
  }

  private SingletonImmutableBiMap(K singleKey, V singleValue, ImmutableBiMap<V, K> inverse) {
    this.singleKey = singleKey;
