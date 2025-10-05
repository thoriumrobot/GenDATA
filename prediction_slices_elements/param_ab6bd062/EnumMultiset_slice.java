// Source-based slice around line 284
// Method: <com.google.common.collect.EnumMultiset: void forEachEntry(ObjIntConsumer)>

          public int getCount() {
            return counts[index];
          }
        };
      }
    };
  }

  @Override
  public void forEachEntry(ObjIntConsumer<? super E> action) {
    checkNotNull(action);
    for (int i = 0; i < enumConstants.length; i++) {
      if (counts[i] > 0) {
        action.accept(enumConstants[i], counts[i]);
      }
    }
  }

  @Override
  public Iterator<E> iterator() {
