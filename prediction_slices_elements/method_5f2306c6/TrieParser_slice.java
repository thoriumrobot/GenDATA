// Source-based slice around line 116
// Method: <com.google.thirdparty.publicsuffix.TrieParser: CharSequence reverse(CharSequence)>

          break;
        }
      }
    }

    stack.pop();
    return idx - start;
  }

  private static CharSequence reverse(CharSequence s) {
    return new StringBuilder(s).reverse();
  }

  private TrieParser() {}
}
