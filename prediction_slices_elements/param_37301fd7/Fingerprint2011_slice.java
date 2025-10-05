// Source-based slice around line 148
// Method: <com.google.common.hash.Fingerprint2011: long hashLength33To64(byte[],int,int)>

      long tmp = z;
      z = x;
      x = tmp;
      offset += 64;
      length -= 64;
    } while (length != 0);
    return hash128to64(hash128to64(v[0], w[0]) + shiftMix(y) * K1 + z, hash128to64(v[1], w[1]) + x);
  }

  private static long hashLength33To64(byte[] bytes, int offset, int length) {
    long z = load64(bytes, offset + 24);
    long a = load64(bytes, offset) + (length + load64(bytes, offset + length - 16)) * K0;
    long b = rotateRight(a + z, 52);
    long c = rotateRight(a, 37);
    a += load64(bytes, offset + 8);
    c += rotateRight(a, 7);
    a += load64(bytes, offset + 16);
    long vf = a + z;
    long vs = b + rotateRight(a, 31) + c;
    a = load64(bytes, offset + 16) + load64(bytes, offset + length - 32);
