// Source-based slice around line 415
// Method: <com.google.common.graph.DirectedGraphConnections: V value(N)>

          }
        }
        return endOfData();
      }
    };
  }

  @SuppressWarnings("unchecked")
  @Override
  public @Nullable V value(N node) {
    checkNotNull(node);
    Object value = adjacentNodeValues.get(node);
    if (value == PRED) {
      return null;
    }
    if (value instanceof PredAndSucc) {
      return (V) ((PredAndSucc) value).successorValue;
    }
    return (V) value;
  }
