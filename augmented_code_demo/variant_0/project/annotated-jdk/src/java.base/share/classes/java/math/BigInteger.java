/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.math;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.common.value.qual.PolyValue;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Random;
    @Positive
import java.util.concurrent.ThreadLocalRandom;
    @Positive
import jdk.internal.math.DoubleConsts;
    @Positive
import jdk.internal.math.FloatConsts;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import jdk.internal.vm.annotation.Stable;

    @Positive
@AnnotatedFor({ "nullness", "value" })
    @Positive
public class BigInteger extends Number implements Comparable<BigInteger> {

    @Positive
    public BigInteger(byte[] val, int off, int len) {
    @Positive
    }

    @Positive
    public BigInteger(byte[] val) {
    @Positive
    }

    @Positive
    public BigInteger(@IntRange(from = -1, to = 1) int signum, byte[] magnitude, int off, int len) {
    @Positive
    }

    @Positive
    public BigInteger(@IntRange(from = -1, to = 1) int signum, byte[] magnitude) {
    @Positive
    }

    @Positive
    public BigInteger(String val, @IntRange(from = 2, to = 36) int radix) {
    @Positive
    }

    @Positive
    public BigInteger(String val) {
    @Positive
    }

    @Positive
    public BigInteger(int numBits, Random rnd) {
    @Positive
    }

    @Positive
    public BigInteger(int bitLength, int certainty, Random rnd) {
    @Positive
    }

    @Positive
    public static BigInteger probablePrime(int bitLength, Random rnd);

    @Positive
    public BigInteger nextProbablePrime();

    @Positive
    boolean primeToCertainty(int certainty, Random random);

    @Positive
    public static BigInteger valueOf(long val);

    @Positive
    public static final BigInteger ZERO;

    @Positive
    public static final BigInteger ONE;

    @Positive
    public static final BigInteger TWO;

    @Positive
    public static final BigInteger TEN;

    @Positive
    public BigInteger add(BigInteger val);

    @Positive
    BigInteger add(long val);

    @Positive
    public BigInteger subtract(BigInteger val);

    @Positive
    public BigInteger multiply(BigInteger val);

    @Positive
    BigInteger multiply(long v);

    @Positive
    public BigInteger divide(BigInteger val);

    @Positive
    public BigInteger[] divideAndRemainder(BigInteger val);

    @Positive
    public BigInteger remainder(BigInteger val);

    @Positive
    public BigInteger pow(int exponent);

    @Positive
    public BigInteger sqrt();

    @Positive
    public BigInteger[] sqrtAndRemainder();

    @Positive
    public BigInteger gcd(BigInteger val);

    @Positive
    static int bitLengthForInt(int n);

    @Positive
    static void primitiveRightShift(int[] a, int len, int n);

    @Positive
    static void primitiveLeftShift(int[] a, int len, int n);

    @Positive
    public BigInteger abs();

    @Positive
    public BigInteger negate();

    @Positive
    @IntRange(from = -1, to = 1)
    @Positive
    public int signum();

    @Positive
    public BigInteger mod(BigInteger m);

    @Positive
    public BigInteger modPow(BigInteger exponent, BigInteger m);

    @Positive
    static int mulAdd(int[] out, int[] in, int offset, int len, int k);

    @Positive
    static int addOne(int[] a, int offset, int mlen, int carry);

    @Positive
    public BigInteger modInverse(BigInteger m);

    @Positive
    public BigInteger shiftLeft(int n);

    @Positive
    public BigInteger shiftRight(int n);

    @Positive
    int[] javaIncrement(int[] val);

    @Positive
    public BigInteger and(BigInteger val);

    @Positive
    public BigInteger or(BigInteger val);

    @Positive
    public BigInteger xor(BigInteger val);

    @Positive
    public BigInteger not();

    @Positive
    public BigInteger andNot(BigInteger val);

    @Positive
    public boolean testBit(int n);

    @Positive
    public BigInteger setBit(int n);

    @Positive
    public BigInteger clearBit(int n);

    @Positive
    public BigInteger flipBit(int n);

    @Positive
    public int getLowestSetBit();

    @Positive
    public int bitLength();

    @Positive
    public int bitCount();

    @Positive
    public boolean isProbablePrime(int certainty);

    @Positive
    @IntRange(from = -1, to = 1)
    @Positive
    public int compareTo(BigInteger val);

    @Positive
    @IntRange(from = -1, to = 1)
    @Positive
    final int compareMagnitude(BigInteger val);

    @Positive
    @IntRange(from = -1, to = 1)
    @Positive
    final int compareMagnitude(long val);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object x);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public BigInteger min(BigInteger val);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public BigInteger max(BigInteger val);

    @Positive
    public int hashCode();

    @Positive
    public String toString(@IntRange(from = 2, to = 36) int radix);

    @Positive
    public String toString();

    @Positive
    public byte[] toByteArray();

    @Positive
    @PolyValue
    @Positive
    public int intValue(@PolyValue BigInteger this);

    @Positive
    @PolyValue
    @Positive
    public long longValue(@PolyValue BigInteger this);

    @Positive
    @PolyValue
    @Positive
    public float floatValue(@PolyValue BigInteger this);

    @Positive
    @PolyValue
    @Positive
    public double doubleValue(@PolyValue BigInteger this);

    @Positive
    private static class UnsafeHolder {

    @Positive
        static void putSign(BigInteger bi, int sign);

    @Positive
        static void putMag(BigInteger bi, int[] magnitude);
    @Positive
    }

    @Positive
    public long longValueExact();

    @Positive
    public int intValueExact();

    @Positive
    public short shortValueExact();

    @Positive
    public byte byteValueExact();
    @Positive
}

// CFWR semantic augmentation - variant 0
