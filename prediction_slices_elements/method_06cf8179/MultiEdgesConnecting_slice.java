// Source-based slice around line 66
// Method: <com.google.common.graph.MultiEdgesConnecting: boolean contains(Object)>

            return entry.getKey();
          }
        }
        return endOfData();
      }
    };
  }

  @Override
  public boolean contains(@Nullable Object edge) {
    return targetNode.equals(outEdgeToNode.get(edge));
  }
}
