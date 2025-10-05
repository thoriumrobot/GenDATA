// Source-based slice around line 54
// Method: com.google.common.collect.ImmutableEnumMap.delegate

        return ImmutableMap.of();
      case 1:
        Entry<K, V> entry = getOnlyElement(map.entrySet());
        return ImmutableMap.of(entry.getKey(), entry.getValue());
      default:
        return new ImmutableEnumMap<>(map);
    }
  }

  private final transient EnumMap<K, V> delegate;

  private ImmutableEnumMap(EnumMap<K, V> delegate) {
    this.delegate = delegate;
    checkArgument(!delegate.isEmpty());
  }

  @Override
  UnmodifiableIterator<K> keyIterator() {
    return Iterators.unmodifiableIterator(delegate.keySet().iterator());
  }
