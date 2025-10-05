// Source-based slice around line 2425
// Method: <com.google.common.collect.MapMakerInternalMap: V put(K,V)>

        break;
      }
      last = sum;
    }
    return false;
  }

  @CanIgnoreReturnValue
  @Override
  public @Nullable V put(K key, V value) {
    checkNotNull(key);
    checkNotNull(value);
    int hash = hash(key);
    return segmentFor(hash).put(key, hash, value, false);
  }

  @CanIgnoreReturnValue
  @Override
  public @Nullable V putIfAbsent(K key, V value) {
    checkNotNull(key);
