// Source-based slice around line 38
// Method: com.google.common.collect.ImmutableMapValues.map


/**
 * {@code values()} implementation for {@link ImmutableMap}.
 *
 * @author Jesse Wilson
 * @author Kevin Bourrillion
 */
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

