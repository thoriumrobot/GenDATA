// Source-based slice around line 695
// Method: <com.google.common.collect.ImmutableMultimap: Spliterator entrySpliterator()>

         * requireNonNull is safe: The first call to this method always enters the !hasNext() case
         * and populates currentKey, after which it's never cleared.
         */
        return immutableEntry(requireNonNull(currentKey), valueItr.next());
      }
    };
  }

  @Override
  Spliterator<Entry<K, V>> entrySpliterator() {
    return CollectSpliterators.flatMap(
        asMap().entrySet().spliterator(),
        keyToValueCollectionEntry -> {
          K key = keyToValueCollectionEntry.getKey();
          Collection<V> valueCollection = keyToValueCollectionEntry.getValue();
          return CollectSpliterators.map(
              valueCollection.spliterator(), (V value) -> immutableEntry(key, value));
        },
        Spliterator.SIZED | (this instanceof SetMultimap ? Spliterator.DISTINCT : 0),
        size());
