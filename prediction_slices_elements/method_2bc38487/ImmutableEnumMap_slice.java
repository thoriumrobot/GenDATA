// Source-based slice around line 42
// Method: <com.google.common.collect.ImmutableEnumMap: ImmutableMap asImmutable(EnumMap)>


/**
 * Implementation of {@link ImmutableMap} backed by a non-empty {@link java.util.EnumMap}.
 *
 * @author Louis Wasserman
 */
@GwtCompatible
@SuppressWarnings("serial") // we're overriding default serialization
final class ImmutableEnumMap<K extends Enum<K>, V> extends IteratorBasedImmutableMap<K, V> {
  static <K extends Enum<K>, V> ImmutableMap<K, V> asImmutable(EnumMap<K, V> map) {
    switch (map.size()) {
      case 0:
        return ImmutableMap.of();
      case 1:
        Entry<K, V> entry = getOnlyElement(map.entrySet());
        return ImmutableMap.of(entry.getKey(), entry.getValue());
      default:
        return new ImmutableEnumMap<>(map);
    }
  }
