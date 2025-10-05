// Source-based slice around line 709
// Method: <com.google.common.collect.ImmutableMultimap: void forEach(BiConsumer)>

          Collection<V> valueCollection = keyToValueCollectionEntry.getValue();
          return CollectSpliterators.map(
              valueCollection.spliterator(), (V value) -> immutableEntry(key, value));
        },
        Spliterator.SIZED | (this instanceof SetMultimap ? Spliterator.DISTINCT : 0),
        size());
  }

  @Override
  public void forEach(BiConsumer<? super K, ? super V> action) {
    checkNotNull(action);
    asMap()
        .forEach(
            (key, valueCollection) -> valueCollection.forEach(value -> action.accept(key, value)));
  }

  /**
   * Returns an immutable multiset containing all the keys in this multimap, in the same order and
   * with the same frequencies as they appear in this multimap; to get only a single occurrence of
   * each key, use {@link #keySet}.
