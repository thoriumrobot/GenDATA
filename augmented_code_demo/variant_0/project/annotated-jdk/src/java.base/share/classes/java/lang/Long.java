/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyIndex;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.lock.qual.NewObject;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.checker.signedness.qual.SignednessGlb;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.checker.signedness.qual.Unsigned;
    @Positive
import org.checkerframework.common.value.qual.ArrayLenRange;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.common.value.qual.IntVal;
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
import java.lang.annotation.Native;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.lang.constant.ConstantDesc;
    @Positive
import java.math.*;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import jdk.internal.misc.CDS;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import static java.lang.String.COMPACT_STRINGS;
    @Positive
import static java.lang.String.LATIN1;
    @Positive
import static java.lang.String.UTF16;

    @Positive
@AnnotatedFor({ "nullness", "index", "signedness", "value" })
    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Long extends Number implements Comparable<Long>, Constable, ConstantDesc {

    @Positive
    @Native
    @Positive
    @SignednessGlb
    @Positive
    @IntVal(0x8000000000000000L)
    @Positive
    public static final long MIN_VALUE;

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    @IntVal(0x7fffffffffffffffL)
    @Positive
    public static final long MAX_VALUE;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static final Class<Long> TYPE;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1)
    @Positive
    public static String toString(long i, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1)
    @Positive
    public static String toUnsignedString(@Unsigned long i, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1, to = 16)
    @Positive
    public static String toHexString(@UnknownSignedness long i);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1, to = 22)
    @Positive
    public static String toOctalString(@Unsigned long i);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1, to = 64)
    @Positive
    public static String toBinaryString(@Unsigned long i);

    @Positive
    static String toUnsignedString0(@Unsigned long val, @IntVal({ 1, 2, 3, 4, 5 }) int shift);

    @Positive
    static String fastUUID(long lsb, long msb);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1, to = 20)
    @Positive
    public static String toString(long i);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String toUnsignedString(@Unsigned long i);

    @Positive
    static int getChars(long i, int index, byte[] buf);

    @Positive
    static int stringSize(long x);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long parseLong(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long parseLong(CharSequence s, int beginIndex, int endIndex, @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long parseLong(String s) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static long parseUnsignedLong(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static long parseUnsignedLong(CharSequence s, int beginIndex, int endIndex, @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static long parseUnsignedLong(String s) throws NumberFormatException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Long valueOf(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Long valueOf(String s) throws NumberFormatException;

    @Positive
    private static class LongCache {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @NewObject
    @Positive
    @PolySigned
    @Positive
    @PolyValue
    @Positive
    public static Long valueOf(@PolySigned @PolyValue long l);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static Long decode(String nm) throws NumberFormatException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    @PolyIndex
    @Positive
    @PolySigned
    @Positive
    @PolyValue
    @Positive
    public Long(@PolyIndex @PolySigned @PolyValue long value) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public Long(String s) throws NumberFormatException {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyIndex
    @Positive
    @PolyValue
    @Positive
    public byte byteValue(@PolyIndex @PolyValue Long this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyIndex
    @Positive
    @PolyValue
    @Positive
    public short shortValue(@PolyIndex @PolyValue Long this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyIndex
    @Positive
    @PolyValue
    @Positive
    public int intValue(@PolyIndex @PolyValue Long this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyIndex
    @Positive
    @PolySigned
    @Positive
    @PolyValue
    @Positive
    public long longValue(@PolyIndex @PolySigned @PolyValue Long this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public float floatValue(@PolyValue Long this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public double doubleValue(@PolyValue Long this);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1, to = 20)
    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int hashCode(long value);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Nullable
    @Positive
    public static Long getLong(@Nullable String nm);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static Long getLong(@Nullable String nm, long val);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyNull
    @Positive
    public static Long getLong(@Nullable String nm, @PolyNull Long val);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int compareTo(Long anotherLong);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compare(long x, long y);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compareUnsigned(@Unsigned long x, @Unsigned long y);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static long divideUnsigned(@Unsigned long dividend, @Unsigned long divisor);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static long remainderUnsigned(@Unsigned long dividend, @Unsigned long divisor);

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    @IntVal(64)
    @Positive
    public static final int SIZE;

    @Positive
    @SignedPositive
    @Positive
    @IntVal(8)
    @Positive
    public static final int BYTES;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long highestOneBit(@UnknownSignedness long i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long lowestOneBit(@UnknownSignedness long i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @NonNegative
    @Positive
    @IntRange(from = 0, to = 64)
    @Positive
    public static int numberOfLeadingZeros(@UnknownSignedness long i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @NonNegative
    @Positive
    @IntRange(from = 0, to = 64)
    @Positive
    public static int numberOfTrailingZeros(@UnknownSignedness long i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @NonNegative
    @Positive
    public static int bitCount(@UnknownSignedness long i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolySigned
    @Positive
    public static long rotateLeft(@PolySigned long i, int distance);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolySigned
    @Positive
    public static long rotateRight(@PolySigned long i, int distance);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @SignednessGlb
    @Positive
    public static long reverse(@PolySigned long i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @GTENegativeOne
    @Positive
    public static int signum(long i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @SignednessGlb
    @Positive
    public static long reverseBytes(@PolySigned long i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long sum(long a, long b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long max(long a, long b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long min(long a, long b);

    @Positive
    @Override
    @Positive
    public Optional<Long> describeConstable();

    @Positive
    @Override
    @Positive
    public Long resolveConstantDesc(MethodHandles.Lookup lookup);
    @Positive
}

// CFWR semantic augmentation - variant 0
